SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32

CHAIN_BREAK_STR = "|"

SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_EOS_STR = "<eos>"

MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"

# fmt: off
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on
