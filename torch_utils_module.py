from torch.utils.data import Dataset
from torch import nn
from tqdm import tqdm
import linecache
import re
import torch

class CustomDatasetForTextVectorizer(Dataset):
    def __init__(self, file_path, random_data=False, min_seq_len=0, max_seq_len=64):
        """
        Initializes the object with the provided file path and optional parameters.

        Parameters:
            file_path (str): The path to the file to be read.
            random_data (bool): Whether to shuffle the lines of the file randomly. Defaults to False.
            min_seq_len (int): The minimum length of the sequences to be generated. Defaults to 0.
            max_seq_len (int): The maximum length of the sequences to be generated. Defaults to 64.

        Returns:
            None
        """
        self.file_path = file_path
        self.data = open(self.file_path, "r").readlines()
        self.random_data = random_data
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pattern = re.compile(r"[^a-zA-Z0-9\s,.\"']")

    def __len__(self):
        """
        Returns the length of the object.

        :return: The length of the object.
        :rtype: int
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve an item from the data at the given index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing two strings. The first string is a substring of the data starting from a random position and ending at `self.max_seq_len - 1` characters. The second string is a substring of the data starting from the next character after the start of the first substring and ending at `self.max_seq_len` characters.

        Raises:
            Exception: If an error occurs while retrieving the item.
        """
        try:
            return self.data[idx].strip().lower()
        except Exception as e:
            print("error on index ", idx)
            print(self.data[idx])
            print(e)


class LazyCustomDataset(Dataset):
    def __init__(self, file_path, random_data=False, min_seq_len=0, max_seq_len=64):
        self.file_path = file_path
        self.random_data = random_data
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pattern = re.compile(r"[^a-zA-Z0-9\s,.\"']")
        self.X = ""
        self.X_len = 0
        self.rand_idx = 0
        self.total_len = len(open(self.file_path, "r").readlines())
    def __len__(self):
        return self.total_len
    def __getitem__(self, idx):
        try:
            self.X = linecache.getline(self.file_path, idx+1)
            self.X = self.pattern.sub("", self.X)
            self.X = self.X.split(" ")
            self.X_len = len(self.X)
            if len(self.X) <= self.max_seq_len:
                self.rand_idx = torch.randint(0, self.X_len, (1,))
                return " ".join(self.X[:self.rand_idx]), " "+self.X[self.rand_idx]+" "
            else:
                self.rand_idx = torch.randint(0, self.X_len - self.max_seq_len, (1,))
                #return (" ".join(self.X[self.rand_idx:self.rand_idx+self.max_seq_len]), nltk.pos_tag(self.X[self.rand_idx:self.rand_idx+self.max_seq_len])), " "+self.X[self.rand_idx+self.max_seq_len]+" "
                return " ".join(self.X[self.rand_idx:self.rand_idx+self.max_seq_len]), " "+self.X[self.rand_idx+self.max_seq_len]+" "
            
        except Exception as e:
            print("error on index ", idx)
            print(linecache.getline(self.file_path, idx+1))
            print(e)


class CustomDataset(Dataset):
    def __init__(self, file_path, random_data=False, min_seq_len=0, max_seq_len=64):
        """
        Initializes the object with the provided file path and optional parameters.

        Parameters:
            file_path (str): The path to the file to be read.
            random_data (bool): Whether to shuffle the lines of the file randomly. Defaults to False.
            min_seq_len (int): The minimum length of the sequences to be generated. Defaults to 0.
            max_seq_len (int): The maximum length of the sequences to be generated. Defaults to 64.

        Returns:
            None
        """
        self.file_path = file_path
        self.data = open(self.file_path, "r").readlines()
        self.random_data = random_data
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pattern = re.compile(r"[^a-zA-Z0-9\s,.\"']")

    def __len__(self):
        """
        Returns the length of the object.

        :return: The length of the object.
        :rtype: int
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve an item from the data at the given index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing two strings. The first string is a substring of the data starting from a random position and ending at `self.max_seq_len - 1` characters. The second string is a substring of the data starting from the next character after the start of the first substring and ending at `self.max_seq_len` characters.

        Raises:
            Exception: If an error occurs while retrieving the item.
        """
        try:
            X = self.data[idx].strip().lower()
            X = self.pattern.sub("", X)
            if len(X) > self.max_seq_len:
                length = len(X)
                random_start = torch.randint(0, length - self.max_seq_len, (1,))
                return X[random_start:random_start+self.max_seq_len-1], X[random_start+1:random_start+self.max_seq_len]
            else:
                return X[0:len(X)-1], X[1:len(X)]
        except Exception as e:
            print("error on index ", idx)
            print(self.data[idx])
            print(e)


class TextVectorizer(nn.Module):
    """
    Text vectorization module in PyTorch.
    """
    def __init__(self, max_tokens = None, split = None, lower = True, strip_punctuation = True):
        """
        Initialize the TextVectorization module.
        
        Args:
            max_tokens (int, optional): Maximum number of tokens. Defaults to None.
            standardize (str, optional): Standardization method. Defaults to 'lower'.
            split (str, optional): Token split method. Defaults to None.
        """
        super(TextVectorizer, self).__init__()
        self.max_tokens = max_tokens
        self.split = split
        self.lower = lower
        self.strip_punctuation = strip_punctuation
        self.vocabulary = dict({"<pad>": 0, "<unk>": 1})
        self.vocabulary_count = dict()
        self.pattern = re.compile(r"[^a-z0-9\s,.\"']")
        self.pattern2 = re.compile(r"[^a-z0-9\s]")

    
    def transform_text(self, text: str):
        if self.lower:
            text = text.lower()
        if self.strip_punctuation:
            text = self.pattern2.sub("", text)
        else:
            text = self.pattern.sub("", text)
        if self.split is not None:
            text = text.split(self.split)
        return text
    
    def adapt(self, dataset, on_labels=True, stop_at_index=None):
        """
        Adapt the vocabulary based on input texts.
        
        Args:
            texts (list): List of input texts.
            on_labels (bool, optional): Whether to adapt on labels or not. Defaults to True.
            stop_at_index (int, optional): Stop adapting at this index. Defaults to None.
        """
        for i, text in tqdm(enumerate(dataset), desc="Adapting vocabulary", total=stop_at_index if stop_at_index is not None else len(dataset)):
            if stop_at_index is not None and i >= stop_at_index:
                break
            text_temp = self.transform_text(text)
            for token in text_temp:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                    self.vocabulary_count[token] = 0
                else:
                    self.vocabulary_count[token] += 1
    
    def prune_vocab(self, threshold):
        """
        Prune the vocabulary based on a threshold.
        
        Args:
            threshold (int): Threshold to prune the vocabulary.
        """
        print("removing low count words")
        print("count threshold: ", threshold)
        print("current vocab size: ", len(self.vocabulary))
        low_count_words = [w for w,c in self.vocabulary_count.items() if c < threshold]
        for w in tqdm(low_count_words):
            del self.vocabulary[w]
            del self.vocabulary_count[w]
        print("new vocab size: ", len(self.vocabulary))
        self.vocabulary = {token: i for i, token in enumerate(self.vocabulary.keys())}
    
    def get_vocabulary(self):
        """
        Get the vocabulary.
        
        Returns:
            vocabulary (dict): Vocabulary.
        """
        return self.vocabulary, self.vocabulary_count
    
    def set_vocabulary(self, vocabulary):
        """
        Set the vocabulary.

        Args:
            vocabulary (dict): Vocabulary.
        """
        self.vocabulary = vocabulary
    
    def load_vocabulary(self, path):
        """
        Load the vocabulary from a file.

        Args:
            path (str): Path to load the vocabulary.
        """
        with open(path, 'rb') as f:
            self.vocabulary, self.vocabulary_count = torch.load(f)

    def save_vocabulary(self, path):
        """
        Save the vocabulary to a file.

        Args:
            path (str): Path to save the vocabulary.
        """
        with open(path, 'wb') as f:
            torch.save([self.vocabulary, self.vocabulary_count], f)

    def forward(self, texts):
        """
        Forward pass of the module.
        
        Args:
            text (str): Input text.
            
        Returns:
            transformed_text (torch.Tensor): Transformed text.
        """
        vectorized_tokens = []
        for text in texts:
            text = self.transform_text(text)
            #transformed_text = torch.zeros(len(text), dtype=torch.int, requires_grad=False)
            """
            Syntax : Dict.get(key, default=None)

            Parameters: 

            key: The key name of the item you want to return the value from
            Value: (Optional) Value to be returned if the key is not found. The default value is None.
            Returns: Returns the value of the item with the specified key or the default value.
            """
            transformed_text = torch.tensor([self.vocabulary.get(token, self.vocabulary["<unk>"]) for token in text], dtype=torch.int, requires_grad=False)
            vectorized_tokens.append(transformed_text)
        return vectorized_tokens


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # Compute focal loss
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None and type(self.alpha) == list:
            focal_loss *= self.alpha[targets]
        elif self.alpha is not None:
            focal_loss *= self.alpha

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")


class Model(nn.Module):
    def __init__(self, in_embedding_dim, pretrained_embedding_path=None, out_embedding_dim=None, rnn_dim=None, num_rnn_layers=0, rnn_dropout=0, bidirectional=False, dense_dims=[], vocab_size=0, mode="char"):
        """
        Initializes the Model class.

        Parameters:
            embedding_dim (int): The dimension of the embedding layer.
            rnn_dim (int): The dimension of the RNN layer.
            num_layers (int): The number of layers in the RNN.
            bidirectional (bool): Whether the RNN is bidirectional or not.
            dense_dims (List[int]): A list of integers representing the dimensions of the dense layers.
            vocab_size (int): The size of the vocabulary.
            mode (str, optional): The mode of the model. Defaults to "char".

        Returns:
            None
        """
        super(Model, self).__init__()

        self.mode = mode
        self.rnn_dim = rnn_dim
        if pretrained_embedding_path is not None:
            weights = torch.load(pretrained_embedding_path)["weight"]
            self.embedding = nn.Embedding(vocab_size, in_embedding_dim, _weight=weights, _freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, in_embedding_dim, padding_idx=0)
        
        if self.rnn_dim is not None:
            self.rnn = nn.LSTM(in_embedding_dim, rnn_dim, num_layers=num_rnn_layers, batch_first=True, bidirectional=bidirectional, dropout=rnn_dropout)

        if mode == "char":
            self.dense_dims = []
            _ = 2 if bidirectional else 1
            dense_start_size = rnn_dim*_ if rnn_dim is not None else in_embedding_dim
            self.dense_dims.append(dense_start_size)
            for dense_dim in dense_dims:
                self.dense_dims.append(dense_dim)
            self.dense_dims.append(vocab_size)
            self.dense = nn.ModuleList([nn.Linear(self.dense_dims[i], self.dense_dims[i+1]) for i in range(len(self.dense_dims)-1)])
        else:
            self.dense_dims = []
            _ = 2 if bidirectional else 1
            dense_start_size = rnn_dim*_ if rnn_dim is not None else in_embedding_dim
            self.dense_dims.append(dense_start_size)
            for dense_dim in dense_dims:
                self.dense_dims.append(dense_dim)
            self.dense_dims.append(out_embedding_dim)
            self.dense = nn.ModuleList([nn.Linear(self.dense_dims[i], self.dense_dims[i+1]) for i in range(len(self.dense_dims)-1)])           


    def forward(self, x: torch.Tensor, word_embed_info = None, state = "inference", debug = False, dropout_allowance=None):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            state (str, optional): The state of the model. Defaults to "inference".
            debug (bool, optional): Whether to print debug information. Defaults to False.
            dropout_allowance (float, optional): The dropout allowance for randomness of output. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of the model.

        Raises:
            Exception: If an error occurs during the forward pass.
        """
        shapes = dict({})
        try:
            shapes["input"] = (x.shape)

            x = self.embedding(x)
            shapes["embedding"] = (x.shape)

            offset = 0
            if word_embed_info is not None:
                word_embed_info = word_embed_info.unsqueeze(-2) if word_embed_info.dim() != x.dim() else word_embed_info
                shapes["embedding_concat"] = (word_embed_info.shape, x.shape)
                offset = word_embed_info.shape[-2]
                x = torch.cat((word_embed_info, x), -2)      # -3, -2, -1 => [batch, seq_len, embedding_dim]
            shapes["embedding-2"] = (x.shape)
            
            
            if self.rnn_dim is not None:
                x , _ = self.rnn(x)
                shapes["rnn-x"] = (x.shape)
                shapes["rnn-_"] = [i.shape for i in _]

            if self.mode == "char":
                if state == "inference":  
                    x = x[:, -1, :] if x.dim() == 3 else x[-1, :]
                elif state == "training":
                    x = x[:, offset:, :] if x.dim() == 3 else x[offset:, :]
                for layer in self.dense:
                    if dropout_allowance:
                        x = nn.Dropout(dropout_allowance)(x)
                    x = nn.Tanh()(x)
                    x = layer(x)
                shapes["dense"] = x.shape

            else:
                if dropout_allowance:
                    x = nn.Dropout(dropout_allowance)(x)
                for layer in self.dense:
                    x = nn.Tanh()(x)
                    x = layer(x)
                    
                shapes["dense"] = x.shape
                #x = torch.sum(x, -2)
                #shapes["sum"] = x.shape
            
            if debug:
                print("shapes:", shapes)

            return x
        except:
            print("\nerror")
            print("shapes:", shapes)
            print("----------------------------------------------------------")
            a = 10 / 0

class Char_word_embed(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_dim, rnn_layers):
        super(Char_word_embed, self).__init__()
        self.char_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.compressor = nn.GRU(embedding_dim, rnn_dim, batch_first=True, num_layers=rnn_layers)
    def forward(self, x):
        x = self.char_embed(x)
        x, _ = self.compressor(x)
        return x[:, -1]

class WordModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_dim, rnn_layers, dense_dims=[]):
        super(WordModel, self).__init__()
        self.char_to_word = Char_word_embed(embedding_dim, vocab_size, rnn_dim, rnn_layers)
        self.dense_dims = []
        self.dense_dims.append(embedding_dim)
        for dense_dim in dense_dims:
            self.dense_dims.append(dense_dim)
        self.dense = nn.ModuleList([nn.Linear(self.dense_dims[i], self.dense_dims[i+1]) for i in range(len(self.dense_dims)-1)])
    def forward(self, x, state = "inference", debug = False, dropout_allowance=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        x = torch.stack(list(map(lambda i: self.char_to_word(i), x))).squeeze(1)
        if state == "inference":  
            x = x.squeeze(0)
        for layer in self.dense:
            x = layer(x)
        return x