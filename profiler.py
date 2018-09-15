import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import time, sys
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class profiler(QtWidgets.QWidget):

	def __init__(self):
		#connects and initializes methods in a parent class
		super(profiler, self).__init__()
		self.cap = cv.VideoCapture(0) # 0  here lets python know which camera we are using
		self.initializeGUI()


	def initializeGUI(self):
		self.setWindowTitle('Laser Beam Profiler')
		layout = QtWidgets.QGridLayout()


		self.setupPlot()
		self.canvasrow = FigureCanvas(self.figurerow)
		self.canvascolumn = FigureCanvas(self.figurecolumn)

		self.xwaist = QtWidgets.QLabel()
		self.ywaist = QtWidgets.QLabel()
		self.xwaist.setStyleSheet('color: #FF6600; font-weight: bold; font-family: Copperplate / Copperplate Gothic Light, sans-serif')
		self.ywaist.setStyleSheet('color: #FF6600; font-weight: bold; font-family: Copperplate / Copperplate Gothic Light, sans-serif')

		layout.addWidget(self.canvasrow)
		layout.addWidget(self.canvascolumn)
		layout.addWidget(self.xwaist, 2,1)
		layout.addWidget(self.ywaist, 3,1)

		self.setLayout(layout)


	def setupPlot(self):
		self.figurerow, self.axrow = plt.subplots()
		self.figurecolumn, self.axcolumn = plt.subplots()

		# create line objects for fast plot redrawing
		self.linesrow, = self.axrow.plot([],[],linewidth=2,color='purple')
		self.linescolumn, = self.axcolumn.plot([],[],linewidth=2,color='blue')

		self.axrow.set_xlim(0,500)
		self.axrow.set_ylim(0,1000)

		self.axcolumn.set_xlim(0,500)
		self.axcolumn.set_ylim(0,1000)


	def start(self):

		while True:

			_,frame = self.cap.read() #cv2.VideoCapture.read() returns a tuple (return value, image)
			#since I am not using the return value anywhere, I use '_' this line to ignore

			# collecting array of red pixels
			redimage = frame[:,:,2] # OpenCV uses BGR convention so 2 is used for red
			globmax = np.max(redimage)

			# total number of pixels in each axis
			xpixels = np.linspace(0,len(redimage[0,:]),len(redimage[0,:]))
			ypixels = np.linspace(0,len(redimage[:,0]),len(redimage[:,0]))

			# convert to one row and one column through summation for live plots
			convrow = redimage.sum(axis=0)/48.0
			convcolumn = redimage.sum(axis=1)/48.0

			# removing background noise
			convrow = convrow - np.min(convrow)
			convcolumn = convcolumn - np.min(convcolumn)

			# initial guess for fitting
			rowampguess = convrow.max()
			rowcenterguess = np.argmax(convrow)

			columnampguess = convcolumn.max()
			columncenterguess = np.argmax(convcolumn[::-1])

			# curve fit convrow and convcolumn to gaussian, fit parameters returned in popt1 and popt2
			try:
				popt1, pcov1 = curve_fit(self.func, xpixels, convrow, p0=[rowampguess, rowcenterguess, 200])
				popt2, pcov2 = curve_fit(self.func, ypixels, convcolumn, p0=[columnampguess, columncenterguess, 200])

			except:
				popt1, popt2 = [[0,0,1], [0,0,1]]

			# updates for live plotting row and column
			self.linesrow.set_xdata(xpixels)
			self.linesrow.set_ydata(convrow[::-1]) #[::-1] is used to invert convrow for better user experience

			self.linescolumn.set_xdata(ypixels)
			self.linescolumn.set_ydata(convcolumn[::-1])

			# draw and flush data
			self.figurerow.canvas.draw()
			self.figurerow.canvas.flush_events()

			self.figurecolumn.canvas.draw()
			self.figurecolumn.canvas.flush_events()

			# update X and Y waist labels with scaled waists
			self.xwaist.setText('X = ' + str(np.abs(popt1[2]*2*5.875))[0:5] + 'um')
			self.ywaist.setText('Y = ' + str(np.abs(popt2[2]*2*5.875))[0:5] + 'um')


	def func(self,x,a,x0, sigma):
		return a*np.exp(-(x-x0)**2/(2*sigma**2))


if __name__ == "__main__":
	app = QtWidgets.QApplication([])
	prof = profiler()
	prof.show()
	prof.start()
