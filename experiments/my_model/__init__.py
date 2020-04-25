"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import openprotein
from preprocessing import process_raw_data
from training import train_model
from torch.nn.utils.rnn import pad_sequence

from util import get_backbone_positions_from_angles, contruct_dataloader_from_disk, initial_pos_from_aa_string, pass_messages, write_out, calc_avg_drmsd_over_minibatch

ANGLE_ARR = torch.tensor([[-120, 140, -370], [0, 120, -150], [25, -120, 150]]).float()

def run_experiment(parser, use_gpu):
    # parse experiment specific command line arguments
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.01, help='Learning rate to use during training.')

    parser.add_argument('--input-file', dest='input_file', type=str,
                        default='data/preprocessed/protein_net_testfile.txt.hdf5')

    args, _unknown = parser.parse_known_args()

    # pre-process data
    process_raw_data(use_gpu, force_pre_processing_overwrite=False)

    # run experiment
    training_file = args.input_file
    validation_file = args.input_file

    model = MyModel(21, use_gpu=use_gpu)  # embed size = 21

    train_loader = contruct_dataloader_from_disk(training_file, args.minibatch_size)
    validation_loader = contruct_dataloader_from_disk(validation_file, args.minibatch_size)

    train_model_path = train_model(data_set_identifier="TRAIN",
                                   model=model,
                                   train_loader=train_loader,
                                   validation_loader=validation_loader,
                                   learning_rate=args.learning_rate,
                                   minibatch_size=args.minibatch_size,
                                   eval_interval=args.eval_interval,
                                   hide_ui=args.hide_ui,
                                   use_gpu=use_gpu,
                                   minimum_updates=args.minimum_updates)

    print("Completed training, trained model stored at:")
    print(train_model_path)

class MyModel(openprotein.BaseModel):
    def __init__(self, embedding_size, use_gpu):
        super(MyModel, self).__init__(use_gpu, embedding_size)
        self.use_gpu = use_gpu
        #RMSD layer
        self.recurrent_steps = 2
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9  # 3 dimensions * 3 coordinates for each aa
        self.f_to_hid = nn.Linear((embedding_size * 2 + 9), self.hidden_size, bias=True)
        self.hid_to_pos = nn.Linear(self.hidden_size, self.msg_output_size, bias=True)
        # (last state + orginal state)
        self.linear_transform = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True)
        #Angles layer
        self.number_angles = 3
        self.input_to_angles = nn.Linear(embedding_size, self.number_angles)

    def apply_message_function(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        aa_features_transformed = torch.cat(
            (
                aa_features[:, 0, 0:21],
                aa_features[:, 1, 0:21],
                aa_features[:, 0, 21:30] - aa_features[:, 1, 21:30]
            ), dim=1)
        return self.hid_to_pos(self.f_to_hid(aa_features_transformed))

    def initial_pos_from_aa_string(batch_aa_string, use_gpu):
        arr_of_angles = []
        batch_sizes = []
        for aa_string in batch_aa_string:
            length_of_protein = aa_string.size(0)
            angles = torch.stack([-120 * torch.ones(length_of_protein),
                                  140 * torch.ones(length_of_protein),
                                  -370 * torch.ones(length_of_protein)]).transpose(0, 1)
            arr_of_angles.append(angles)
            batch_sizes.append(length_of_protein)

        padded = pad_sequence(arr_of_angles).transpose(0, 1)
        return get_backbone_positions_from_angles(padded, batch_sizes, use_gpu)

    def pass_messages(aa_features, message_transformation, use_gpu):
        # aa_features (#aa, #features) - each row represents the amino acid type
        # (embedding) and the positions of the backbone atoms
        # message_transformation: (-1 * 2 * feature_size) -> (-1 * output message size)
        feature_size = aa_features.size(1)
        aa_count = aa_features.size(0)

        arange2d = torch.arange(aa_count).repeat(aa_count).view((aa_count, aa_count))

        diagonal_matrix = (arange2d == arange2d.transpose(0, 1)).int()

        eye = diagonal_matrix.view(-1).expand(2, feature_size, -1) \
            .transpose(1, 2).transpose(0, 1)

        eye_inverted = torch.ones(eye.size(), dtype=torch.uint8) - eye
        if use_gpu:
            eye_inverted = eye_inverted.cuda()
        features_repeated = aa_features.repeat((aa_count, 1)).view((aa_count, aa_count, feature_size))
        # (aa_count^2 - aa_count) x 2 x aa_features     (all pairs except for reflexive connections)
        aa_messages = torch.stack((features_repeated.transpose(0, 1), features_repeated)) \
            .transpose(0, 1).transpose(1, 2).view(-1, 2, feature_size)

        eye_inverted_location = eye_inverted.view(-1).nonzero().squeeze(1)

        aa_msg_pairs = aa_messages \
            .reshape(-1).gather(0, eye_inverted_location).view(-1, 2, feature_size)

        transformed = message_transformation(aa_msg_pairs).view(aa_count, aa_count - 1, -1)
        transformed_sum = transformed.sum(dim=1)  # aa_count x output message size
        return transformed_sum

    def _get_network_emissions(self, original_aa_string):
        #RMSD
        backbone_atoms_padded, batch_sizes_backbone = initial_pos_from_aa_string(original_aa_string, self.use_gpu)
        embedding_padded = self.embed(original_aa_string)

        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()

        for _ in range(self.recurrent_steps):
            combined_features = torch.cat(
                (embedding_padded, backbone_atoms_padded),
                dim=2
            ).transpose(0, 1)

            features_transformed = []

            for aa_features in combined_features.split(1, dim=0):
                msg = pass_messages(aa_features.squeeze(0),
                                    self.apply_message_function,
                                    self.use_gpu)  # aa_count * output size
                features_transformed.append(self.linear_transform(
                    torch.cat((aa_features.squeeze(0), msg), dim=1)))

            backbone_atoms_padded_clone = torch.stack(features_transformed).transpose(0, 1)

        backbone_atoms_padded = backbone_atoms_padded_clone

        #return [], backbone_atoms_padded, batch_sizes_backbone

        #ANGLES

        batch_sizes = list([a.size() for a in original_aa_string])

        #embedded_input = self.embed(original_aa_string)
        emissions_padded = self.input_to_angles(embedding_padded)

        probabilities = torch.softmax(emissions_padded.transpose(0, 1), 2)

        output_angles = torch.matmul(probabilities, ANGLE_ARR).transpose(0, 1)

        print(batch_sizes)
        print(batch_sizes_backbone)

        return output_angles, backbone_atoms_padded, batch_sizes
