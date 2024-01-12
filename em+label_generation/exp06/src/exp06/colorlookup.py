import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import math

def color_swatch(
    colors,
    edgecolors=None,
    show_text=False,
    text_threshold=0.6,
    ax=None,
    title=None,
    one_row=None,
):
    """
    Display the colours defined in a list of colors.

    :param colors: List of (r,g,b) colour tuples to display. (r,g,b) should be floats
        between 0 and 1.

    :param edgecolors: If None displayed colours have no outline. Otherwise a list of
        (r,g,b) colours to use as outlines for each colour.

    :param show_text: If True writes the background colour's hex on top of it in black
        or white, as appropriate.

    :param text_threshold: float between 0 and 1. With threshold close to 1 white text
        will be chosen more often.

    :param ax: Matplotlib axis to plot to. If ax is None plt.show() is run in function
        call.

    :param title: Add a title to the colour swatch.

    :param one_row: If True display colours on one row, if False as a grid. If
        one_row=None a grid is used when there are more than 8 colours.

    :return:
    """
    import matplotlib.colors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if one_row is None:
        if len(colors) > 8:
            one_row = False
        else:
            one_row = True

    if one_row:
        n_grid = len(colors)
    else:
        n_grid = math.ceil(np.sqrt(len(colors)))

    width = 1
    height = 1

    x = 0
    y = 0

    max_x = 0
    max_y = 0

    if ax is None:
        show = True
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect="equal")
    else:
        show = False

    for idx, (cls, color) in enumerate(colors.items()):
        if edgecolors is None:
            ax.add_patch(patches.Rectangle((x, y), width, height, color=color))
        else:
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=edgecolors[idx],
                    linewidth=5,
                )
            )

        if show_text:
            ax.text(
                x + (width / 2),
                y + (height / 2),
                cls,
                fontsize=60 / np.sqrt(len(colors)),
                ha="center",
                color=distinctipy.get_text_color(color, threshold=text_threshold),
            )

        if (idx + 1) % n_grid == 0:
            if edgecolors is None:
                y += height
                x = 0
            else:
                y += height + (height / 10)
                x = 0
        else:
            if edgecolors is None:
                x += width
            else:
                x += width + (width / 10)

        if x > max_x:
            max_x = x

        if y > max_y:
            max_y = y

    ax.set_ylim([-height / 10, max_y + 1.1 * height])
    ax.set_xlim([-width / 10, max_x + 1.1 * width])
    ax.invert_yaxis()
    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
        
class_colors =  {
            "ecs": (50,50,50),
            "pm": (100,100,100),
            "mito_mem": (255,128,0),
            "mito_lum": (128,64,0),
            "mito_ribo": (220,172,104),
            "golgi_mem": (0,132,255),
            "golgi_lum": (0,66,128),
            "ves_mem": (255,0,0),
            "ves_lum": (128,0,0),
            "endo_mem": (0,0,255),
            "endo_lum": (0,0,128),
            "lyso_mem": (255,216,0),
            "lyso_lum": (128, 108,0),
            "ld_mem": (134,164,247),
            "ld_lum": (79,66,252),
            "er_mem": (57,215,46),
            "er_lum": (51,128,46),
            "eres_mem": (85,254,219),
            "eres_lum": (6,185,157),
            "ne_mem": (9,128,0),
            "ne_lum": (5,77,0),
            "np_out": (175,249,111),
            "np_in": (252,144,211),
            "hchrom": (168,55,188),
            "nhchrom": (84,23,94),
            "echrom": (204,0,102),
            "nechrom": (102,0,51),
            "nucpl": (255,0,255),
            "nucleo": (247,82,104),
            "mt_out": (255,255,255),
            "mt_in": (128,128,128),}

colors = {k: tuple(np.array(color)/255.) for k, color in class_colors.items()}
color_swatch(colors, show_text=True)
plt.show()