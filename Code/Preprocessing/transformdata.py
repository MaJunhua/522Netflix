from datetime import datetime
import random
import os 
path = "/Users/zhangqi/Google Drive/Study/CSC 522/522 Project/data"
os.chdir(path)

# Tansform the original data format to 'movie user rating timestamp'
# Need to put all the original data into path folder and each file needs end with .txt
def transformData(path):
	files = os.listdir(path)
	with open("combinedData_total.data", 'wb') as output:
		for f in files:
			if f.endswith('.txt'):
				with open(f) as readinput:
					lines = readinput.readlines()
	    			for line in lines:
	    				if ":" in line:
	    					movie = line.split(":")[0]
	    				else:
	    					[user,rate,date]= line.split(',')
	    					date = date.split('\n')[0]
	    					date = datetime.strptime(date, "%Y-%m-%d")
	    					date = date.strftime('%s')
	    					newline = movie + "," + user + "," + rate + "," + date + "\n"
	    					output.write(newline)
	    					# print newline
	    					# break
	    			readinput.close()
			# break
		output.close()

# random sample from the original data set
def randomSampleData(path):
	with open("combinedData_01percent.data", 'wb') as output:
		with open("combinedData_total.data", 'r') as readinput:
			print 'Reading data...'
			lines = readinput.readlines()
			print 'Read data complete. Starting sampling...'
			sampledata = random.sample(lines[:100*len(lines)/1000],1*len(lines)/1000)
			print 'Sample complete. Staring writing to file...'
			for line in sampledata:
				output.write(line)
			readinput.close()
		output.close()
	print 'Writing complete. Closed all files.'

# stratify sample from the original data set with shrink users and movies
def StratifySampleData(path, user_boundry, movie_boundry):
	with open("movieID_lessthan500.data", 'wb') as output:
		with open("combinedData_total.data", 'r') as readinput:
			print 'Reading data...'
			lines = readinput.readlines()
			print 'Read data complete. Starting sampling...'
			i = j = 0
			for line in lines:
				movie, user, rating, time = line.split(',')
				if int(movie) < movie_boundry and int(user) < user_boundry:
					output.write(line)
					j += 1
				i += 1
				if i % 1000000 == 0:
					print 'processing:', float(i)*100 / len(lines), '%'
			readinput.close()
		output.close()
	print 'Writing complete. Closed all files. Samples = ', j

StratifySampleData(path,200000,500)

# sampleData(path)



# tranferData(path)