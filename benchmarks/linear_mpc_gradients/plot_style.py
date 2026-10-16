"""Shared figure styling for the sensitivity accuracy study."""

PAPER_FONT_SIZE = 4
OURS_BLUE_DARK = "#08519C"
PAPER_LINE_ALPHA = 0.6

def _apply_paper_rc(plt, font_size: int = PAPER_FONT_SIZE) -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
            "legend.frameon": False,
        }
    )


def _paper_title(text: str) -> str:
    return rf"\textbf{{{text}}}"


