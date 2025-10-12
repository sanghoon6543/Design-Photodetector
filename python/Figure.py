
from tkinter import *
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import UI_Config as UI
import DrawingUtility as UTIL


fd = pathlib.Path(__file__).parent.resolve()
fh = 400
fw = fh*2 + 100
fs = (fw/200, 0.7*fh/100)

class DeviceModelling:
    def __init__(self, window):
        self.window = UI.UI_tkinter.MakeWindow(window, 'Device Modelling', fw, fh, False, background='grey94')
        self.__main__()
        self.InputInfo = self.EntryAddress.copy()

    def Event_ApplyInfo(self, frame, entryadress, ftstyle='Calibri', ftsize=24, tickftsize=10):

        ### Update Ipuntinfo
        for key in entryadress:
            self.InputInfo[key] = UTIL.DataProcessing.GetEntry(entryadress[key])

        ## Make Preview Widget
        if not hasattr(self, 'ax'):
            self.ax, self.canvas = UTIL.FigureConfig.MakeFigureWidget(frame, (0.6*fs[0], 0.95*fs[1]))

        ### Draw Preview Figure
        self.ax.cla()
        UTIL.FigureConfig.FigureConfiguration(self.ax, self.InputInfo, ftstyle, ftsize, tickftsize)
        UTIL.FigureConfig.forceAspect(self.ax, self.InputInfo['xScale'], self.InputInfo['yScale'], aspect=1)
        self.canvas.draw()
        plt.close(plt.gcf())

    def Event_NewFigure(self, inputinfo, ftstyle='Calibri', ftsize=24, tickftsize=10):
        fig, self.drawax = plt.subplots(figsize=fs, tight_layout=True)

        UTIL.FigureConfig.FigureConfiguration(self.drawax, inputinfo, ftstyle, ftsize, tickftsize)
        UTIL.FigureConfig.forceAspect(self.drawax, inputinfo['xScale'], inputinfo['yScale'], aspect=1)

    def Event_DrawClipboard(self, ax, color, legendinfo, alphavalue, marker='None', ftsize=16):

        data = UTIL.DataProcessing.ReadClipboard()
        c = next(color)

        ax.plot(data[0], data[1], marker, c=c, alpha=alphavalue)

        ax.legend(legendinfo[:], loc='best', fontsize=ftsize)
        plt.pause(0.001)

    def Event_Calculate(self, ax, inputinfo, ftsize=16):
        VStep, dStep, N, mt, alpha = float(inputinfo['xStep']), float(inputinfo['yStep']), int(inputinfo['N']), float(inputinfo['mutau']), float(inputinfo['abscoeff'])
        colorstyle = mpl.colormaps[inputinfo['CMapTitleLd_0']]
        vmin, vmax = float(inputinfo['CMap Range_0']), float(inputinfo['CMap Range_1'])
        V = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + VStep, VStep)
        d = UTIL.DataProcessing.um2cm(np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + dStep, dStep))

        data = UTIL.DataProcessing.HechtRelation(V, d, mt, alpha, N)
        c = ax.imshow(data, cmap=colorstyle, alpha=0.6, origin='lower' , vmin = vmin, vmax = vmax)

        ax.cbar = ax.get_figure().colorbar(c, ax=ax)
        ax.cbar.set_label(label=inputinfo['CMapTitleLd_1'], size=ftsize)
        ax.cbar.ax.tick_params(labelsize=ftsize)

        if bool(inputinfo['CMapTitleLd_2']) == True:
            UTIL.DataProcessing.drawSchubweg(ax, V, mt)

        plt.pause(0.001)
        plt.show()


    def __main__(self):
        self.InputInfoFrame = UI.UI_tkinter.MakeFrameLabel(self.window, fw, fh, 0, 0, "Plot Configuration")
        self.OutputFrame = UI.UI_tkinter.MakeFrameLabel(self.window, fw, fh, 1, 0, "Figure Preview")
        self.DataProcessFrame = UI.UI_tkinter.MakeFrameLabel(self.window, fw, fh, 2, 0, "Data Processing")
        ### Input UI

        colspan = 0

        LabelInfos = ["Title", "x-Axis Title", "y-Axis Title", "Applied Bias [V]", "Thickness [\u03BCm]", "MajorTick X Y",
                      'CMap Range', "CMap Title Ld"]
        for n, t in enumerate(LabelInfos):
            UI.UI_tkinter.UI_Labels(self.InputInfoFrame, t=t, row=n)

        colspan += 3

        EntryInfos = {'Title': 'Title', 'xAxisTitle': "Applied Bias, V [V]", 'yAxisTitle': "Thickness, d [\u03BCm]",
                      'xLim': (0, 1), 'yLim': (0, 1), 'MajorTickXY': (1, 1), 'CMap Range': (0, 1), 'CMapTitleLd': ('RdBu_r', 'Intensity [a.u.]', True)}

        self.EntryAddress = {}

        for k, key in enumerate(EntryInfos):
            if type(EntryInfos[key]) is tuple:
                n = EntryInfos[key].__len__()
                for t1, tt in enumerate(EntryInfos[key]):
                    self.EntryAddress[key + f'_{t1}'] = UI.UI_tkinter.UI_InputEntry(self.InputInfoFrame, tt, row=k, col=1+t1, width=6)
            else:
                self.EntryAddress[key] = UI.UI_tkinter.UI_InputEntry(self.InputInfoFrame, EntryInfos[key], row=k, col=1, colspan=colspan, width=24)

        CBoxInfos = {'xScale': ["Linear", "SymLog"], 'yScale': ["Linear", "SymLog"], 'Grid': ["Grid ON", "Grid Off"]}
        for k, key in enumerate(CBoxInfos):
            self.EntryAddress[key] = UI.UI_tkinter.UI_CBox(self.InputInfoFrame, CBoxInfos[key], row=k+3, col=3, width=6, padx=1, pady=1, ftsize=8)

        colspan += 1

        ButtonInfos = ['ApplyInfo']
        self.ButtonAddress = {}
        for n, t in enumerate(ButtonInfos):
            self.ButtonAddress[t] = UI.UI_tkinter.UI_Button(self.InputInfoFrame, t, row=LabelInfos.__len__(), col=0, colspan=colspan, width=30, height=1)

        ### Output UI
        colspan = 0

        self.OutputPlotFrame = UI.UI_tkinter.MakeFrame(self.OutputFrame, 60*fs[0], 95*fs[1], 0, 0, 3)

        colspan += 1
        ButtonInfos = ["New Figure"]
        for n, t in enumerate(ButtonInfos):
            self.ButtonAddress[t] = UI.UI_tkinter.UI_Button(self.OutputFrame, t, row=1, col=0, colspan=colspan, width=20, height=1)

        ### Data Processing UI
        colspan = 0
        LabelInfos = ["x Step", "y Step", 'N', "\u03BC\u03C4 [cm\u207b\u00B9]", "\u03B1 [cm\u00b2V\u207b\u00B9s\u207b\u00B9]"]

        colspan += 1
        for n, t in enumerate(LabelInfos):
            UI.UI_tkinter.UI_Labels(self.DataProcessFrame, t=t, row=n)

        EntryInfos = {'xStep': 1, 'yStep': 1, 'N': 1000, 'mutau': 1E-7, 'abscoeff': 476}
        for k, key in enumerate(EntryInfos):
                self.EntryAddress[key] = UI.UI_tkinter.UI_InputEntry(self.DataProcessFrame, EntryInfos[key], row=k, col=1, colspan=colspan, width=10)

        colspan += 1
        ButtonInfos = ["Calculate"]
        for n, t in enumerate(ButtonInfos):
            self.ButtonAddress[t] = UI.UI_tkinter.UI_Button(self.DataProcessFrame, t, row=LabelInfos.__len__(), col=0, colspan=colspan, width=20, height=1)


        ### Designate Button Callback Function
        self.ButtonAddress['ApplyInfo'].configure(command=lambda: self.Event_ApplyInfo(self.OutputPlotFrame, self.EntryAddress))
        self.ButtonAddress['New Figure'].configure(command=lambda: self.Event_NewFigure(self.InputInfo.copy()))
        self.ButtonAddress['Calculate'].configure(command=lambda: self.Event_Calculate(self.drawax, self.InputInfo.copy()))

if __name__ == '__main__':
    window = Tk()
    DeviceModelling(window)
    window.mainloop()