import numpy as np
import pandas as pd

import xlsxwriter

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import ticker
from matplotlib.colors import SymLogNorm
import matplotlib.image as mImage

import tkinter
from tkinter import *
import pathlib
from os.path import isfile, join
'''''''''''''''''''''''''''''''''''''''''''''''''''Data Export'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#x1 = np.arange(-1000,1000,0.2)
#d1 = data[:, 3000]
#c1 = np.array([x1,d1]).T
#df = pd.DataFrame(c1)
#df.to_clipboard(excel=True, sep=None, index=False, header=None)
#Paste and Copy Excel and Paste to ExceltoFig Gaussian Fit.py
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

colorstyle = plt.cm.RdBu_r
alphavalue = 1

fd = pathlib.Path(__file__).parent.resolve()
tkfont = {'Font': 'Calibri', 'FontSize': 10}
tickfontstyle = {'Font': 'Calibri', 'FontSize': 18}
fontstyle = {'Font': 'Calibri', 'FontSize': 24}
DefaultInfos = {'Title': 'Title', 'xAxisTitle': "Applied Bias, V [V]", 'yAxisTitle': "Thickness, d [\u03BCm]",
                'xLim': (0, 1), 'yLim': (0, 1), 'MajorTickX': 1, 'MajorTickY': 1, 'CMapMin': 0, 'CMapMax': 1, 'CMapTitle': 'Intensity [a. u.]'}

fh = 400
fw = fh*2
fs = (fw/200, 0.92*0.7*fh/100)


class GaussianBeamPath:
    def __init__(self, window):
        self.window = window
        self.window.title("Gaussian Beam Propagation Calculation")
        # self.window.config(background='#FFFFFF')
        self.window.geometry(f"{fw}x{fh}")
        self.window.resizable(False, False)

        self.filepath = ""

        self.__main__()

    def __main__(self):

        self.InputInfoFrame = LabelFrame(self.window, width=fw, height=fh, text="Plot 속성", font=(f"{fontstyle['Font']} {fontstyle['FontSize']}"))
        self.InputInfoFrame.grid(column=0, row=0, padx=10, pady=10)

        self.OutputFrame = LabelFrame(self.window, width=fw, height=fh, text="Figure Property Preview", font=(f"{fontstyle['Font']} {fontstyle['FontSize']}"))
        self.OutputFrame.grid(column=1, row=0, padx=10, pady=10)

        self.OutputPlotFrame = Frame(self.OutputFrame, bg='white', width=100*fs[0], height=90*fs[1])
        self.OutputPlotFrame.grid(column=0, row=0, columnspan=4, padx=10, pady=10)

        TitleLable = Label(self.InputInfoFrame, width=14, height=2, text="Title", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        TitleLable.grid(row=0, column=0, padx=2, pady=2)
        xAxisLable = Label(self.InputInfoFrame, width=14, height=2, text="x-Axis Title", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        xAxisLable.grid(row=1, column=0, padx=2, pady=2)
        yAxisLable = Label(self.InputInfoFrame, width=14, height=2, text="y-Axis Title", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        yAxisLable.grid(row=2, column=0, padx=2, pady=2)
        xLimLable = Label(self.InputInfoFrame, width=14, height=2, text="Applied Bias [V]", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        xLimLable.grid(row=3, column=0, padx=2, pady=2)
        yLimLable = Label(self.InputInfoFrame, width=14, height=2, text="Thickness [\u03BCm]", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        yLimLable.grid(row=4, column=0, padx=2, pady=2)
        MajorTickLable = Label(self.InputInfoFrame, width=14, height=2, text="MajorTick X Y", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        MajorTickLable.grid(row=5, column=0, padx=2, pady=2)
        ApplyInfo = Button(self.InputInfoFrame, width=14, height=2, text="Apply", relief="raised", font=(f"{tkfont['Font']} {tkfont['FontSize']}"), command=self.Applyinfo)
        ApplyInfo.grid(row=6, column=0, columnspan=3, padx=2, pady=2)

        self.TitleEntry = Entry(self.InputInfoFrame, width=20, relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.TitleEntry.grid(row=0, column=1, columnspan=2, padx=2, pady=2)
        self.TitleEntry.insert(0, DefaultInfos['Title'])

        self.xAxisEntry = Entry(self.InputInfoFrame, width=20, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.xAxisEntry.grid(row=1, column=1, columnspan=2, padx=2, pady=2)
        self.xAxisEntry.insert(0, DefaultInfos['xAxisTitle'])

        self.yAxisEntry = Entry(self.InputInfoFrame, width=20, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.yAxisEntry.grid(row=2, column=1, columnspan=2, padx=2, pady=2)
        self.yAxisEntry.insert(0, DefaultInfos['yAxisTitle'])

        self.xLimEntryDN = Entry(self.InputInfoFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.xLimEntryDN.grid(row=3, column=1, padx=2, pady=2)
        self.xLimEntryDN.insert(0, DefaultInfos['xLim'][0])

        self.xLimEntryUP = Entry(self.InputInfoFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.xLimEntryUP.grid(row=3, column=2, padx=2, pady=2)
        self.xLimEntryUP.insert(0, DefaultInfos['xLim'][1])

        self.yLimEntryDN = Entry(self.InputInfoFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.yLimEntryDN.grid(row=4, column=1, padx=2, pady=2)
        self.yLimEntryDN.insert(0, DefaultInfos['yLim'][0])

        self.yLimEntryUP = Entry(self.InputInfoFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.yLimEntryUP.grid(row=4, column=2, padx=2, pady=2)
        self.yLimEntryUP.insert(0, DefaultInfos['yLim'][1])

        self.MajorTickEntryX = Entry(self.InputInfoFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.MajorTickEntryX.grid(row=5, column=1, padx=2, pady=2)
        self.MajorTickEntryX.insert(0, DefaultInfos['MajorTickX'])

        self.MajorTickEntryY = Entry(self.InputInfoFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.MajorTickEntryY.grid(row=5, column=2, padx=2, pady=2)
        self.MajorTickEntryY.insert(0, DefaultInfos['MajorTickY'])

        xStepLable = Label(self.OutputFrame, width=14, text="x Step", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        xStepLable.grid(row=1, column=0, padx=1, pady=1)

        yStepLable = Label(self.OutputFrame, width=14, text="y Step", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        yStepLable.grid(row=2, column=0, padx=1, pady=1)

        mutauLabel = Label(self.OutputFrame, width=14, text="\u03BC\u03C4 [cm\u207b\u00B9]", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        mutauLabel.grid(row=1, column=2, padx=1, pady=1)

        NLabel = Label(self.OutputFrame, width=14, text="N", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        NLabel.grid(row=2, column=2, padx=1, pady=1)

        alphaLabel = Label(self.OutputFrame, width=14, text="\u03B1 [cm\u00b2V\u207b\u00B9s\u207b\u00B9]", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        alphaLabel.grid(row=3, column=0, padx=1, pady=1)


        self.xStepVal = Entry(self.OutputFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.xStepVal.grid(row=1, column=1, padx=1, pady=1)
        self.xStepVal.insert(0, 1)

        self.yStepVal = Entry(self.OutputFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.yStepVal.grid(row=2, column=1, padx=1, pady=1)
        self.yStepVal.insert(0, 1)

        self.mutau = Entry(self.OutputFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.mutau.grid(row=1, column=3, padx=1, pady=1)
        self.mutau.insert(0, 1E-7)

        self.N = Entry(self.OutputFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.N.grid(row=2, column=3, padx=1, pady=1)
        self.N.insert(0, 100)

        self.alpha = Entry(self.OutputFrame, width=8, textvariable="", relief="ridge", font=(f"{tkfont['Font']} {tkfont['FontSize']}"))
        self.alpha.grid(row=3, column=1, padx=1, pady=1)
        self.alpha.insert(0, 476)

        NewFigure = Button(self.OutputFrame, width=20, height=1, text="Draw Efficiency", relief="raised", font=(f"{tkfont['Font']} {tkfont['FontSize']}"), command=self.NewFig)
        NewFigure.grid(row=3, column=2, columnspan=2, padx=1, pady=1)


        # SaveFigure = Button(self.OutputFrame, width=14, height=2, text="Save Figure", relief="raised", font=(f"{tkfont['Font']} {tkfont['FontSize']}"), command=self.SaveFigure)
        # SaveFigure.grid(row=1, column=2, padx=2, pady=2)

    def Applyinfo(self):

        self.UpdateInfos()

        if not hasattr(self, 'ax'):
            self.MakeFigure()

        self.ax.cla()
        self.FigureOptionSetting(self.ax)

    def UpdateInfos(self):
        DefaultInfos['Title'] = self.TitleEntry.get()
        # DefaultInfos['xAxisTitle'] = self.xAxisEntry.get()
        # DefaultInfos['yAxisTitle'] = self.yAxisEntry.get()
        DefaultInfos['xLim'] = (float(self.xLimEntryDN.get()), float(self.xLimEntryUP.get()))
        DefaultInfos['yLim'] = (float(self.yLimEntryDN.get()), float(self.yLimEntryUP.get()))
        DefaultInfos['MajorTickX'] = float(self.MajorTickEntryX.get())
        DefaultInfos['MajorTickY'] = float(self.MajorTickEntryY.get())

    def MakeFigure(self):
        self.fig, self.ax = plt.subplots(figsize=(fs[0], fs[1]))
        self.output_plt = FigureCanvasTkAgg(self.fig, self.OutputPlotFrame)
        self.output_plt.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=1)
        plt.close(self.fig)

    def FigureOptionSetting(self, ax):

        ax.set_title(DefaultInfos['Title'], font=fontstyle['Font'], fontsize=fontstyle['FontSize'])
        ax.set_xlabel(DefaultInfos['xAxisTitle'], font=fontstyle['Font'], fontsize=fontstyle['FontSize'])
        ax.set_ylabel(DefaultInfos['yAxisTitle'], font=fontstyle['Font'], fontsize=fontstyle['FontSize'])
        ax.set_xlim(DefaultInfos['xLim'][0], DefaultInfos['xLim'][1])
        ax.set_ylim(DefaultInfos['yLim'][0], DefaultInfos['yLim'][1])
        ax.xaxis.set_major_locator(MultipleLocator(DefaultInfos['MajorTickX']))
        ax.yaxis.set_major_locator(MultipleLocator(DefaultInfos['MajorTickY']))

        ax.grid(True)
        ax.tick_params(axis='x', labelsize=tickfontstyle['FontSize'])
        ax.tick_params(axis='y', labelsize=tickfontstyle['FontSize'])
        plt.tight_layout()
        # self.forceAspect(ax)

    def DrawFig(self, fig, ax):

        Vstep, dstep, mt, N, alpha = self.getParams()
        V = np.arange(DefaultInfos['xLim'][0], DefaultInfos['xLim'][1] + Vstep, Vstep)
        d = self.um2cm(np.arange(DefaultInfos['yLim'][0], DefaultInfos['yLim'][1] + dstep, dstep))

        data = self.HechtRelation(V, d, mt, alpha, N)
        # data = self.ModifiedHechtRelation(V, d, mt, alpha, N)
        asdf = 1
        linthy, linthx = np.where(data == np.max(data))
        linth = int(np.ceil(np.abs(np.log10(data[linthy[0], -1])))) + 1
        # for k in range(linthx[0]):
        #     data[:, k] = data[:, linthx[0]]

        c = ax.imshow(data, cmap=colorstyle, alpha=alphavalue,
                      extent=[DefaultInfos['xLim'][0], DefaultInfos['xLim'][1],DefaultInfos['yLim'][0], DefaultInfos['yLim'][1]],
                      origin='lower'
                      ,vmin = 0, vmax = 0.5355
                      #,norm = SymLogNorm(linthresh=np.power(10, float(-linth)), vmin=DefaultInfos['CMapMin'],
                      #                   vmax=DefaultInfos['CMapMax'], base=10)
                      )
        # norm = SymLogNorm(linthresh=np.power(10, float(-linth)), vmin=DefaultInfos['CMapMin'],
        #                   vmax=DefaultInfos['CMapMax'], base=10)
        ax.cbar = fig.colorbar(c, ax=ax)
        ax.cbar.set_label(label=DefaultInfos['CMapTitle'], size=fontstyle['FontSize'])
        ax.cbar.ax.tick_params(labelsize=tickfontstyle['FontSize'])
        # ax.cbar.locator = ticker.MaxNLocator(nbins=linth+1)
        # ax.cbar.set_ticks(np.power(10, -np.linspace(linth, 0, linth+1, endpoint=True)))
        # ax.cbar.update_ticks()

        # ax.text(DefaultInfos['xLim'][0] - DefaultInfos['xLim'][0] / 3, DefaultInfos['yLim'][1] - DefaultInfos['yLim'][1] / 3,
        #         f"NA ={np.round(np.sin(theta), 2)}",  fontsize=fontstyle['FontSize'], bbox={'facecolor':'white', 'alpha':0.2})

        # self.forceAspect(self.drawax)

        self.drawSchubweg(ax, V, mt)

        plt.pause(0.001)
        plt.show()

        asdf = 1

    def NewFig(self):
        fig, self.drawax = plt.subplots(figsize=(fs[0], fs[1]))
        self.FigureOptionSetting(self.drawax)
        # self.forceAspect(self.drawax)
        self.DrawFig(fig, self.drawax)

    def getParams(self):
        xstep = float(self.xStepVal.get())
        ystep = float(self.yStepVal.get())
        mutau = float(self.mutau.get())
        N = int(self.N.get())
        alpha = float(self.alpha.get())

        return xstep, ystep, mutau, N, alpha

    def HechtRelation(self, V, d, mt, alpha, N):
        meshV, meshd = np.meshgrid(V, d)
        s = mt*meshV/meshd # schubweg
        Q = 0

        for k in range(1, N+1):
            Q = Q + (s/meshd) * np.exp(-alpha*meshd*k/N) * (1 - np.exp(-alpha*meshd/N)) * (1 - np.exp(-(meshd-k*meshd/N)/s))

        return Q

    def ModifiedHechtRelation(self, V, d, mt, alpha, N):
        meshV, meshd = np.meshgrid(V, d)
        s = mt*meshV/meshd # schubweg
        Q = 0

        for k in range(1, round((N+1)/2)):

            Q = Q + (s/(meshd/2)) * np.exp(-alpha*meshd*k/N) * (1 - np.exp(-alpha*meshd/N)) * (1 - np.exp(-((meshd/2)-k*meshd/N)/s))

        for k in range(round((N+1)/2), round(N+1)):

            Q = Q + (s/meshd) * np.exp(-alpha*meshd*k/N) * (1 - np.exp(-alpha*meshd/N)) * (1 - np.exp(-(meshd-k*meshd/N)/s))

        return Q


    def drawSchubweg(self, ax, V, mt):
        slimit = self.cm2um(np.sqrt(mt*V))
        ax.plot(V, slimit, 'g')
        return

    def um2cm(self, value):
        return value*1E-4

    def cm2um(self, value):
        return value*1E4

    def GaussainBeamPropagation(self, z, r, theta):

        meshz, meshr = np.meshgrid(z, r)
        wz = meshz * np.tan(theta)

        return np.power(1/wz, 2) * np.exp(-2*np.power(meshr/wz, 2))

    def ReadClipboard(self):

        return pd.read_clipboard(header=None)

    def forceAspect(self, ax, aspect=1):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / aspect)

    def PaintFace(self):
        x = float(self.FaceColorXVal.get())
        self.drawax.axvspan(x-10, x+10, facecolor='g', alpha=0.2)


    def SaveFigure(self):

        filepath = tkinter.filedialog.asksaveasfilename(initialdir=f"{fd}/",
                                                        title="Save as",
                                                        filetypes=(("png", ".png"),
                                                                   ("all files", "*")))
        filepath = f"{filepath}.png"

        self.fig.savefig(filepath)

if __name__ == '__main__':
    window = Tk()
    GaussianBeamPath(window)
    window.mainloop()