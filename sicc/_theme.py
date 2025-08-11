from rich.theme import Theme

theme = Theme(
    {
        #
        "ic10.bool": "yellow",
        "ic10.int": "",
        "ic10.float": "",
        "ic10.str": "green",
        "ic10.undef": "bold reverse red",
        "ic10.other": "magenta red",
        #
        "ic10.opcode": "",
        "ic10.raw_opcode": "green bold",
        "ic10.jump": "cyan bold",
        "ic10.label": "magenta bold italic",
        "ic10.label_private": "magenta italic",
        #
        "ic10.title": "cyan bold",
        "ic10.comment": "white dim italic",
        "ic10.linenum": "white dim",
    }
)
