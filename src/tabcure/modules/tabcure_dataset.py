import random
import typing as tp

from datasets import Dataset


class TabCureDataset(Dataset):
    """TabCure Dataset

    The TabCureDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_tokenizer(self, tokenizer):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        shuffled_text = ", ".join(
            ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
        )

        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)