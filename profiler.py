#This software was inspired from the work of Anthony Ransford at https://github.com/aransfor/PiBeamProfiler.

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL.ImageQt import ImageQt
from scipy.optimize import curve_fit
from scipy.misc.pilutil import toimage
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class profiler(QtWidgets.QWidget):

	def __init__(self):
		#connects and initializes methods in a parent class
		super(profiler, self).__init__()
		#desktop = QtWidgets.QDesktopWidget()
		#screensize = desktop.availableGeometry()
		#self.screenres = [screensize.width(), screensize.height()]

		self.cap = cv.VideoCapture(0) # 0  here lets python know which camera we are using
		self.imageres = [640, 480] # put resolution of the camera used
		self.cap.set(3, self.imageres[0])
		self.cap.set(4, self.imageres[1])
		self.cap.set(5, 33) # 33 Hz framerate
		self.cap.set(15, 500) # shutter speed or exposure
		self.cap.set(20, 300) # iso
		# 3,4 are property identifiers for width and height https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set
		self.initializeGUI()


	def initializeGUI(self):
		self.setWindowTitle('Laser Beam Profiler')
		#self.setGeometry(0, 0, self.screenres[0], self.screenres[1])
		layout = QtWidgets.QGridLayout()


		self.setupPlot()

		self.canvasdisp = FigureCanvas(self.figuredisp)
		self.canvasrow = FigureCanvas(self.figurerow)
		self.canvascolumn = FigureCanvas(self.figurecolumn)

		self.videowindow = QtWidgets.QLabel(self)

		self.xwaist = QtWidgets.QLabel()
		self.ywaist = QtWidgets.QLabel()
		self.xwaist.setStyleSheet('color: #FF6600; font-weight: bold; font-family: Copperplate / Copperplate Gothic Light, sans-serif')
		self.ywaist.setStyleSheet('color: #FF6600; font-weight: bold; font-family: Copperplate / Copperplate Gothic Light, sans-serif')

		#layout.addWidget(self.videowindow, 0,0,2,1)
		layout.addWidget(self.canvasdisp, 0,0,2,1)
		layout.addWidget(self.canvasrow, 2,0,2,1)
		layout.addWidget(self.canvascolumn, 0,1,2,1)
		layout.addWidget(self.xwaist, 2,1)
		layout.addWidget(self.ywaist, 3,1)

		self.setLayout(layout)


	def setupPlot(self):
		self.figuredisp, self.axdisp = plt.subplots()
		self.figurerow, self.axrow = plt.subplots()
		self.figurecolumn, self.axcolumn = plt.subplots()

		# create line objects for fast plot redrawing
		self.linesrow, = self.axrow.plot([],[],linewidth=2,color='purple')
		self.linescolumn, = self.axcolumn.plot([],[],linewidth=2,color='purple')

		self.linesrowfit, = self.axrow.plot([],[],linestyle='--',linewidth=2,color='yellow')
		self.linescolumnfit, = self.axcolumn.plot([],[],linestyle='--',linewidth=2,color='yellow')

		self.axdisp.set_title('Laser Display')
		self.axrow.set_title('Horizontal (x-axis)')
		self.axcolumn.set_title('Vertical (y-axis)')

		self.axdisp.set_xlim(0, self.imageres[0])
		self.axdisp.set_ylim(0, self.imageres[1])

		self.axrow.set_xlim(0, self.imageres[0])
		self.axrow.set_ylim(0,600)

		self.axcolumn.set_xlim(0,600)
		self.axcolumn.set_ylim(0, self.imageres[1])


	def start(self):

		while True:

			_,frame = self.cap.read() #cv2.VideoCapture.read() returns a tuple (return value, image)
			#since I am not using the return value anywhere, I use '_' this line to ignore

			# collecting array of red pixels
			redimage = frame[:,:,0] # OpenCV uses BGR convention so 2 is used for red
			globmax = np.max(redimage)

			# total number of pixels in each axis
			xpixels = np.linspace(0,len(redimage[0,:]),len(redimage[0,:]))
			ypixels = np.linspace(0,len(redimage[:,0]),len(redimage[:,0]))

			# convert to one row and one column through summation for live plots
			convrow = redimage.sum(axis=0)/40.0
			convcolumn = redimage.sum(axis=1)/40.0

			# removing background noise
			convrow = convrow - np.min(convrow)
			convcolumn = convcolumn - np.min(convcolumn)

			# initial guess for fitting
			rowampguess = convrow.max()
			rowcenterguess = np.argmax(convrow)

			columnampguess = convcolumn.max()
			columncenterguess = np.argmax(convcolumn[::-1])

			# initial curve fit convrow and convcolumn to gaussian, fit parameters returned in popt1 and popt2
			try:
				popt1, pcov1 = curve_fit(self.func, xpixels, convrow, p0=[rowampguess, rowcenterguess, 200])
				popt2, pcov2 = curve_fit(self.func, ypixels, convcolumn, p0=[columnampguess, columncenterguess, 200])

			except:
				popt1, popt2 = [[0,0,1], [0,0,1]]

			# updates for live plotting row and column
			self.linesrow.set_xdata(xpixels)
			self.linesrow.set_ydata(convrow)

			self.linescolumn.set_xdata(convcolumn)
			self.linescolumn.set_ydata(ypixels)

			# best value updates for fit of row and column subplots
			self.linesrowfit.set_xdata(xpixels)
			self.linesrowfit.set_ydata(self.func(xpixels, popt1[0],popt1[1],popt1[2]))

			self.linescolumnfit.set_xdata(self.func(ypixels, popt2[0],popt2[1],popt2[2]))
			self.linescolumnfit.set_ydata(ypixels)

			# displaying Laser
			#qPixmap = self.nparrayToQPixmap(frame)
			#videoy = int(self.screenres[0]/2.1)
			#videox = int(1.333 * videoy)
			#self.videowindow.setPixmap(qPixmap.scaled(videox,videoy))

			self.axdisp.imshow(frame)

			self.figuredisp.canvas.draw()
			self.figuredisp.canvas.flush_events()

			# draw and flush data
			self.figurerow.canvas.draw()
			self.figurerow.canvas.flush_events()

			self.figurecolumn.canvas.draw()
			self.figurecolumn.canvas.flush_events()

			# calibrate here
			# update X and Y waist labels with scaled waists
			self.xwaist.setText('X = ' + str(np.abs(popt1[2]*2*5.875))[0:5] + 'um')
			self.ywaist.setText('Y = ' + str(np.abs(popt2[2]*2*5.875))[0:5] + 'um')

			# cv2 break procesure
			if cv.waitKey(10) & 0xFF == ord('q'):
				break

	# Gaussian function
	def func(self,x,a,x0, sigma):
		return a*np.exp(-(x-x0)**2/(2*sigma**2))

	#def nparrayToQPixmap(self, arrayImage):
		#pilImage = toimage(arrayImage)
		#qtImage = ImageQt(pilImage)
		#qImage = QtGui.QImage(qtImage)
		#qPixmap = QtGui.QPixmap(qImage)
		#return qPixmap

if __name__ == "__main__":
	app = QtWidgets.QApplication([])
	prof = profiler()
	prof.show()
	prof.start()
