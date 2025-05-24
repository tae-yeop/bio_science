import numpy as np

class Tokenizer(object):
    special_tokens = ['<CLS>','<BOS>','<EOS>','<PAD>','<MSK>','<UNK>']

    def __init__(self, list_tokens=None, file_name=None):
        with open(file_name, 'r') as f:
            list_tokens = [line.strip() for line in f.readlines()]
        
        for spc in self.special_tokens:
            if spc not in list_tokens:
                list_tokens.append(spc)

        self.tokens = list_tokens
        self.vocab_size = len(self.tokens)
        self.tok2id = dict(zip(self.tokens, range(self.vocab_size)))
        self.id2tok = {v: k for k, v in self.tok2id.items()}

    def have_invalid_token(self, token_list):
        for i, token in enumerate(token_list):
            if token not in self.tok2id.keys():
                return True
        return False
    
    def encode(self, token_list):
        idlist = np.zeros(len(token_list), dtype=np.int32)
        for i, token in enumerate(token_list):
            try:
                idlist[i] = self.tok2id[token]
            except KeyError as err:
                print("encode(): KeyError occurred! %s"%err)
                raise
        return idlist

    def decode(self, idlist):
        return [self.id2tok[i] for i in idlist]
    
    def truncate_eos(self, batch_seqs:np.ndarray):
        """
        This function cuts off the tokens(id form) after the first <EOS> in each sample of batch.
        - Input: batch of token lists np.ndarray(batch_size x seq_len)
        - Output: truncated sequence list
        """
        bs, _ = batch_seqs.shape
        seq_list = []
        for i in range(bs):
            ids = batch_seqs[i].tolist()
            # append EOS at the end
            ids.append(self.get_eosi())
            # find EOS position of first encounter
            EOS_pos = ids.index(self.get_eosi())
            # get the seq until right before EOS
            seq_list.append(ids[0:EOS_pos])
        return seq_list
    
    def locate_specials(self, seq):
        """
        Return special (BOS, EOS, PAD, or any custom special) positions in the token id sequence
        """
        spinds = [self.tok2id[spt] for spt in self.special_tokens]
        special_pos = []
        for i, token in enumerate(seq):
            if token in spinds:
                special_pos.append(i)
        return special_pos
    

    def get_clsi(self): return self.tok2id['<CLS>']
    def get_bosi(self): return self.tok2id['<BOS>']
    def get_eosi(self): return self.tok2id['<EOS>']
    def get_padi(self): return self.tok2id['<PAD>']
    def get_mski(self): return self.tok2id['<MSK>']
    def get_unki(self): return self.tok2id['<UNK>']


class SmilesTokenizer(Tokenizer):
    def __init__(self, list_tokens=None, file_name=None):
        super(SmilesTokenizer, self).__init__(list_tokens, file_name)

        # 두 글자 token (Cl, Br, Si, Na 같은)를 하나로 인식하기 위해
        self.multi_chars = set()
        for token in self.tokens:
            if len(token) >= 2 and token not in self.special_tokens:
                self.multi_chars.add(token)

    def tokenize(self, string):
        """
        Tokenization of string->List(token).
        Note that we expect "string" not to contain any special tokens.

        - Multi-character tokens (e.g., 'Cl', 'Br') are recognized and split accordingly.
        if not string:
            raise ValueError("Input string for tokenization cannot be empty.")
        token_list = [string]
        - If the SMILES string contains unregistered or invalid characters, they will remain in the output as-is.
        """
        # start with splitting
        token_list = [string]
        for k_token in self.multi_chars:
            new_tl = []
            for elem in token_list:
                sub_list = []
                # split the sub smiles with the multi-char token
                splits = elem.split(k_token)
                # sub_list will have multi-char token between each split
                for i in range(len(splits) - 1):
                    sub_list.append(splits[i])
                    sub_list.append(k_token)
                sub_list.append(splits[-1]) 
                new_tl.extend(sub_list)
            token_list = new_tl

        # Now, only one-char tokens to be parsed remain.
        new_tl = []
        for token in token_list:
            if token not in self.multi_chars:
                new_tl.extend(list(token))
            else:
                new_tl.append(token)
        # Note that invalid smiles characters can be produced, if the smiles contains un-registered characters.
        return new_tl