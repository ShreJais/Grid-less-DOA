import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
# warnings.filterwarnings('ignore')

DEFAULT_WIDTH = 6.0
DEFAULT_HEIGHT = 1.5
SIZE_SMALL = 20

# install 
# sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

def latexify(
    width_scale_factor = 1,
    height_scale_factor = 1,
    fig_width = None, 
    fig_height = None,
    font_size = SIZE_SMALL
    ):
    f"""
    : params
        width_scale_factor: float, DEFAULT_WIDTH will be divided by this number, DEFAULT_WIDTH is page width: 
                            {DEFAULT_WIDTH} inches.
        height_scale_factor: float, DEFAULT_HEIGHT will be divided by this number, DEFAULT_WIDTH is page width: 
                            {DEFAULT_HEIGHT} inches.
        fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored).
        fig_height: float, height of the figure in inches (if this is specified, height_scale_factor is ignored).
        font_size: float, font size 
    """
    if "LATEXIFY" not in os.environ:
        warnings.warn("LATEXIFY environment variable not set, not latexifying.")
        return 
    
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor
    if fig_height is None:
        fig_height = DEFAULT_HEIGHT / height_scale_factor
    
    # Use TrueType fonts so they are embedded.
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['lines.markersize'] = 20
    # Font sizes
    # https://stackoverflow.com/a/39566040
    mpl.rc('font', size=font_size) # controls default text size.
    mpl.rc('axes', titlesize=font_size) # fontsize of the axes title.
    mpl.rc('axes', labelsize=font_size) # fontsize of the x and y labels.
    mpl.rc('xtick', labelsize=20) # fontsize of the tick labels.
    mpl.rc('ytick', labelsize=20) # fontsize of the tick labels.
    mpl.rc('legend', fontsize=23) # legend fontsize.
    mpl.rc('figure', titlesize= font_size) # fontsize of the figure title.
    
    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    mpl.rcParams['backend'] = 'ps'
    # mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')
    mpl.rc('figure', figsize=(fig_width, fig_height))

def is_latexify_enabled():
    """
    returns true if LATEXIFY environment variable is set.
    """
    return "LATEXIFY" in os.environ 

def _get_fig_name(fname_full):
    fname_full = fname_full.replace("_latexified", "")
    LATEXIFY="LATEXIFY" in os.environ
    extension = "_latexified.pdf" if LATEXIFY else ".pdf"

    if fname_full[-4:] in [".png", ".pdf", ".jpg"]:
        fname = fname_full[:-4]
        warnings.warn(f'renaming {fname_full} to {fname}{extension} because LATEXIFY is {LATEXIFY}.')
    else:
        fname = fname_full
    return fname + extension

def savefig(
    f_name, 
    tight_layout=True, 
    tight_bbox=False, 
    pad_inches=0.0, *args, **kwargs):
    
    if len(f_name) == 0:
        return
    if "FIG_DIR" not in os.environ:
        warnings.warn("set FIG_DIR environment variable to save figures.")
        return 
    
    fig_dir = os.environ['FIG_DIR']
    # Auto create the directory if it doesn't exist.
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    fname_full = os.path.join(fig_dir, f_name)
    fname_full = _get_fig_name(fname_full=fname_full)

    print(f'Saving image to {fname_full}')

    if tight_layout:
        plt.tight_layout(pad=pad_inches)
    print(f'Figure size: {plt.gcf().get_size_inches()}')

    if tight_bbox:
        # This changes the size of the figure.
        plt.savefig(fname_full, pad_inches=pad_inches, bbox_inches='tight', *args, **kwargs)
    else:
        plt.savefig(fname_full, pad_inches=pad_inches, *args, **kwargs)
    
    if "DUAL_SAVE" in os.environ:
        if fname_full.endswith(".pdf"):
            fname_full = fname_full[:-4] + ".png"
        else:
            fname_full = fname_full[:-4] + ".pdf"
        if tight_bbox:
            # This changes the size of the figure.
            plt.savefig(fname_full, pad_inches=pad_inches, bbox_inches='tight', *args, **kwargs)
        else:
            plt.savefig(fname_full, pad_inches=pad_inches, *args, **kwargs)