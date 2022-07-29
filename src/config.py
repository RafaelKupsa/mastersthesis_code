from transformers import XLMRobertaConfig, RobertaConfig


class XLMRobertaForTLMandALPConfig(XLMRobertaConfig):
    """
    Config class for XLMRoberta for TLM (Translation Language Modeling) and ALP (Alignment Link Prediction)
    """
    def __init__(
        self,
        vocab_size=250001,
        alp_hidden_size=64,
        alp_num_heads=2,
        alp_dropout=0.5,
        **kwargs
    ):
        """
        Initializes the config
        :param vocab_size: vocabulary size of the tokenizer
        :param alp_hidden_size: hidden size of the graph layers
        :param alp_num_heads: number of heads for the graph layers
        :param alp_dropout: dropout for the graph layers
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.alp_hidden_size = alp_hidden_size
        self.alp_num_heads = alp_num_heads
        self.alp_dropout = alp_dropout
        self.vocab_size = vocab_size


class XLMRobertaForALPConfig(RobertaConfig):
    """
    Config class for XLMRoberta for ALP (Alignment Link Prediction)
    """
    def __init__(
        self,
        alp_hidden_size=64,
        alp_num_heads=2,
        alp_dropout=0.5,
        **kwargs
    ):
        """
        Initializes the config
        :param alp_hidden_size: hidden size of the graph layers
        :param alp_num_heads: number of heads for the graph layers
        :param alp_dropout: dropout for the graph layers
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.alp_hidden_size = alp_hidden_size
        self.alp_num_heads = alp_num_heads
        self.alp_dropout = alp_dropout
