import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# from transformers import LongformerSelfAttention

from helpers import d

class AttentionError(Exception):
    pass

class MultiheadedAttention(nn.Module):
    """
    Narrow multiheaded attention. Each attention head inspects a
    fraction of the embedding space and expresses attention vectors for each sequence position as a weighted average of all (earlier) positions.
    """

    def __init__(self, d_model, heads=8, dropout=0.1, relative_pos=True):

        super().__init__()
        if d_model % heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = 1024
        #relative positional embeddings
        self.relative_pos = relative_pos
        if relative_pos:
            self.Er = torch.randn([heads, self.max_length, s],
                    device=d())
        else:
            self.Er = None

    def forward(self, x, mask):
        #batch size, sequence length, embedding dimension
        b, t, e = x.size()
        print(f"b:{b} T:{t} e:{e} ") # batch 16, sequence length 1024, embedding 64
        h = self.heads
        #each head inspects a fraction of the embedded space
        #head dimension
        s = e // h
        #start index of position embedding
        embedding_start = self.max_length - t
        x = x.view(b,t,h,s)
        queries, keys, values = [w(x).transpose(1,2)
                for w, x in zip(self.linears, (x,x,x))]

        print(f' query shape: {queries.shape}')  # B, 8, seq, 8
        print(f' Key shape: {keys.shape}')  # B, 8, seq, 8
        print(f' values shape: {values.shape}') # B, 8, seq, 8

        if self.relative_pos:
            #apply same position embeddings across the batch
            #Is it possible to apply positional self-attention over
            #only half of all relative distances?
            Er  = self.Er[:, embedding_start:, :].unsqueeze(0)
            print(f'ER size : {Er.shape}') # [1, 8, 1024, 8]
            QEr = torch.matmul(queries, Er.transpose(-1,-2))
            print(f'QER size : {QEr.shape}') # torch.Size([16, 8, 1024, 1024]) B = 16

            QEr = self._mask_positions(QEr)
            print(f'QER size after mask: {QEr.shape}') # size doesn't change

            #Get relative position attention scores
            #combine batch with head dimension
            SRel = self._skew(QEr).contiguous().view(b*h, t, t)
            print(f'SRel size : {SRel.shape}')  # 16 * 8 , squence, sequence

        else:
            SRel = torch.zeros([b*h, t, t], device=d())
        queries, keys, values = map(lambda x: x.contiguous()\
                .view(b*h, t, s), (queries, keys, values))
        #Compute scaled dot-product self-attention
        #scale pre-matrix multiplication
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))

        scores = torch.bmm(queries, keys.transpose(1, 2))
        print(f'Scores size : {scores.shape}') # B * heads, seq leng, seq len
        scores = scores + SRel
        print(f'Scores size + SREL : {scores.shape}') # B * heads, seq leng, seq len

        #(b*h, t, t)

        subsequent_mask = torch.triu(torch.ones(1, t, t, device=d()),
                1)
        print(f'subsequent_mask size : {subsequent_mask.shape}') # 1, seq length, seq length,

        scores = scores.masked_fill(subsequent_mask == 1, -1e9)
        print(f' Post mask Scores size: {scores.shape}') # B * heads, seq length, seq length,


        if mask is not None:
            mask = mask.repeat_interleave(h, 0)
            wtf = (mask == 0).nonzero().transpose(0,1)
            scores[wtf[0], wtf[1], :] = -1e9


        #Convert scores to probabilities
        attn_probs = F.softmax(scores, dim=2)
        print(f' attn_probs size: {attn_probs.shape}') # B * heads, seq length, seq length,

        attn_probs = self.dropout(attn_probs)
        print(f' dropout attn_probs size: {attn_probs.shape}') # B * heads, seq length, seq length,

        #use attention to get a weighted average of values
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        print(f'out: {out.shape}') # [B, h, seq_len, h]
        #transpose and recombine attention heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        print(f'out trans: {out.shape}') # [B, seq len, embedding size ie/

        #last linear layer of weights
        outfinal = self.recombine_heads(out)
        print(f'out_final : {outfinal.shape}')
        return outfinal


    def _mask_positions(self, qe):
        #QEr is a matrix of queries (absolute position) dot distance embeddings (relative pos).
        #Mask out invalid relative positions: e.g. if sequence length is L, the query at
        #L-1 can only attend to distance r = 0 (no looking backward).
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=d()), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]


class LongMultiheadedAttention(nn.Module):
    """
    Narrow multiheaded attention. Each attention head inspects a
    fraction of the embedding space and expresses attention vectors for each sequence position as a weighted average of all (earlier) positions.
    """

    def __init__(self, d_model, heads=8, dropout=0.1, relative_pos=False, attention_window=32):

        super().__init__()
        if d_model % heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.one_sided_attn_window_size = attention_window // 2
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = 1024
        # relative positional embeddings
        self.relative_pos = relative_pos
        if relative_pos:
            self.Er = torch.randn([heads, self.max_length, s],
                                  device=d())
        else:
            self.Er = None


    def forward(self, x, mask):
        # batch size, sequence length, embedding dimension
        b, t, e = x.size()
        h = self.heads
        # each head inspects a fraction of the embedded space
        # head dimension
        s = e // h
        # start index of position embedding
        embedding_start = self.max_length - t
        x = x.view(b, t, h, s)
        queries, keys, values = [w(x).transpose(1, 2)
                                 for w, x in zip(self.linears, (x, x, x))]
        if self.relative_pos:
            # apply same position embeddings across the batch
            # Is it possible to apply positional self-attention over
            # only half of all relative distances?
            Er = self.Er[:, embedding_start:, :].unsqueeze(0)
            QEr = torch.matmul(queries, Er.transpose(-1, -2))
            QEr = self._mask_positions(QEr)
            # Get relative position attention scores
            # combine batch with head dimension
            SRel = self._skew(QEr).contiguous().view(b * h, t, t)
        else:
            queries /= math.sqrt(s)
            queries = queries.view(t, b, self.heads, s).transpose(0, 1)
            keys = keys.view(t, b, self.heads, s).transpose(0, 1)
            attn_scores = self._sliding_chunks_query_key_matmul(
                queries, keys, self.one_sided_attn_window_size
            )
            # values to pad for attention probs
            remove_from_windowed_attention_mask = (mask != 0)[:, :, None, None]

            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(queries).masked_fill(
                remove_from_windowed_attention_mask, -10000.0
            )
            # diagonal mask with zeros everywhere and -inf inplace of padding
            diagonal_mask = self._sliding_chunks_query_key_matmul(
                float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
            )
            print(f"diagonal_mask size (SRL) : {diagonal_mask.shape}")

            # pad local attention probs
            attn_scores += diagonal_mask
            print(f"attn_scores size : {attn_scores.shape}")
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = attn_probs.type_as(attn_scores)
            del attn_scores
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            values = values.view(t, b, self.heads, s).transpose(0, 1)
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, values, self.one_sided_attn_window_size
            )
            assert attn_output.size() == (b, t, self.heads, s), "Unexpected size"
            attn_output = attn_output.transpose(0, 1).reshape(t, b, e).contiguous()
            outputs = attn_output.transpose(0, 1)
            print(f"outputs size {outputs.shape}")
            return self.recombine_heads(outputs)

        # Compute scaled dot-product self-attention
        # scale pre-matrix multiplication
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores + SRel
        # (b*h, t, t)

        subsequent_mask = torch.triu(torch.ones(1, t, t, device=d()),
                                     1)
        scores = scores.masked_fill(subsequent_mask == 1, -1e9)
        if mask is not None:
            mask = mask.repeat_interleave(h, 0)
            wtf = (mask == 0).nonzero().transpose(0, 1)
            scores[wtf[0], wtf[1], :] = -1e9

        # Convert scores to probabilities
        attn_probs = F.softmax(scores, dim=2)
        attn_probs = self.dropout(attn_probs)
        # use attention to get a weighted average of values
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        # transpose and recombine attention heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        # last linear layer of weights
        return self.recombine_heads(out)

    def _mask_positions(self, qe):
        # QEr is a matrix of queries (absolute position) dot distance embeddings (relative pos).
        # Mask out invalid relative positions: e.g. if sequence length is L, the query at
        # L-1 can only attend to distance r = 0 (no looking backward).
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=d()), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        # pad a column of zeros on the left
        padded_qe = F.pad(qe, [1, 0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        # take out first (padded) row
        return padded_qe[:, :, 1:, :]

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            hidden_states.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hidden_states.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = F.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores
