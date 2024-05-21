# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: analysis
# ---

from sysvar.corrections import *

a = Correction1D("charged_slow_pi", "MC15ri")
a.save_figures = True
a.register_figure_saving_info(["ch1", "t1", "etc"], "boo", "my_plots", ".svg")
a.plot_error_comparison()

a.figure_save_info


figure_save_info = {
    "namespace": None,
    "top_dir": None,
    "dir_spec": None,
    "extra_ext": None,
}

list(figure_save_info.values())
