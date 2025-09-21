#!/usr/bin/env python
# coding: utf-8
"""
How to Save and Load :math:`S^2` data
============================================

在 ``exmaple 1`` 中我们已经展示了如何通过 ``Generator`` 对象生成时间序列和符号表达式数据，并通过 ``SeriesParams`` 和 ``SymbolParams`` 对象来传入特定的参数来进一步调控其生成过程。

对于生成的 :math:`S^2` 数据，我们同样在 ``utils`` 模块中给出了具体的用于保存和加载 :math:`S^2` 数据的两个函数，并内置在了 ``Generator`` 对象中。
"""

# %%

# Importing data generators, parameter controllers and visualization functions
from S2Generator import Generator, SeriesParams, SymbolParams, plot_series
# sphinx_gallery_thumbnail_path = 'source/_static/background.png'

# %%
