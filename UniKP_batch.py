import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
# build_vocab, pretrain_trfm, utils packages are from SMILES Transformer
from transformers import T5EncoderModel, T5Tokenizer
# transformers package is from ProtTrans
import re
import gc
import numpy as np
import pandas as pd
import pickle
import math
from Bio import SeqIO


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i + 1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


def generate_mutated_sequences(protein_sequence):
    mutated_sequences = []
    positions = []
    mutated_aa = []

    for i in range(len(protein_sequence)):
        original_aa = protein_sequence[i]

        for mutation_aa in "ACDEFGHIKLMNPQRSTVWY":
            if mutation_aa != original_aa:
                mutated_sequence = list(protein_sequence)
                mutated_sequence[i] = mutation_aa
                mutated_sequences.append("".join(mutated_sequence))
                positions.append(i + 1)
                mutated_aa.append(mutation_aa)

    return mutated_sequences, positions, mutated_aa


def Load_fasta_files(file_path):
    sequences = []
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            sequences.append(sequence)
    
    return sequences


if __name__ == '__main__':
    # You can input self-defined sequences with fasta format
    # file_path = "Sequences.fasta"
    # sequences = Load_fasta_files(file_path)

    # You can also input your own protein_sequence and generate all of single site mutations by the generate_mutated_sequences function
    protein_sequence = "XXX"
    sequences, positions, mutated_aa = generate_mutated_sequences(protein_sequence)

    # Input substrates smiles
    Smiles = ['XXX'] * len(sequences)

    seq_vec = Seq_to_vec(sequences)
    smiles_vec = smiles_to_vec(Smiles)
    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)

    # For kcat
    with open('UniKP for kcat.pkl', "rb") as f:
        model = pickle.load(f)

    Pre_label = model.predict(fused_vector)
    Pre_label_pow = [math.pow(10, Pre_label[i]) for i in range(len(Pre_label))]
    print(len(Pre_label))

    # res = pd.DataFrame({'sequences': sequences, 'ids': ids, 'Smiles': Smiles, 'Pre_label': Pre_label_pow})
    # res.to_excel('XXX.xlsx')
