import cv2 as cv
import numpy as np
import msvcrt
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class profiler():
	def __init__(self):
		self.cap = cv.VideoCapture(0)


	def func(self,x,a,x0, sigma):
		return a*np.exp(-(x-x0)**2/(2*sigma**2))
		
		
	def start(self):
		aborted = True
		
		while aborted:
			_,frame = self.cap.read()
			
			# collecting array of red pixels
			redimage = frame[:,:,2]
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
				#print("try")
				
			except:
				popt1, popt2 = [[0,0,1], [0,0,1]]	
				#print("except")
			
			# test
			print(xpixels,convrow,popt1)
			
			# program closing
			k = cv.waitKey(1) & 0xFF
			
			if msvcrt.kbhit() and msvcrt.getch()==chr(27).encode():
				aborted = False
			
			
if __name__ == "__main__":
	prof = profiler()
	prof.start()