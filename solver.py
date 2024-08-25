import pytesseract
import numpy as np 
from PIL import Image
import cv2
import os

class ImageConverter:
    def __init__(self,letters=None,words=None):
        self.letters_image=letters
        self.words_image=words
        self.words=[]
        self.letters=[]
        self.boxes=[]

    #transforming the image into grayscale and then rounding values to either 0 or 255 based on given threshold
    def preprocess_image(self, image, threshold,file_path='') -> cv2.typing.MatLike:
        grayscale=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

        if file_path!='':
            result = Image.fromarray((binary).astype(np.uint8))
            result.save(file_path)

        return binary

    #Based on given paramaters passing image through preprocessing and optionally noise reduction and converting it into a list with words to be found
    def words_image_OCR(self,noise_reduction=True, SENS=3,preprocessing_threshold=150, post_reduction_file_path='') -> None:
        image = self.preprocess_image(self.words_image,preprocessing_threshold,post_reduction_file_path)
        if noise_reduction:
            image = self.noise_reduciton(image, SENS=SENS,file_path=post_reduction_file_path)

        output = pytesseract.image_to_string(image,lang='pol', config=r"-c tessedit_char_blacklist={}[](:;)|\/<>,.-=+_!@#$%^&*1234567890l`")
        words=output.splitlines()
        self.words=[word.replace(' ','') for word in words if word != '']

    #Based on given paramaters passing image through preprocessing and optionally noise reduction and converting it into a matrix of letters.
    def letters_image_OCR(self,noise_reduction=True,SENS=3,preprocessing_threshold=150, post_reduction_file_path='') -> None:
        image = self.preprocess_image(self.letters_image,preprocessing_threshold,post_reduction_file_path)
        if noise_reduction:
            image = self.noise_reduciton(image, SENS,post_reduction_file_path)

        boxes = pytesseract.image_to_boxes(image,lang='pol', config=r"-c tessedit_char_blacklist={}[](:;)|\/<>,.-=+_!@#$%^&*1234567890l` --psm 6")
        for box in boxes.splitlines():
            b = box.split()
            character = b[0]
            left, bottom, right, top = map(int, b[1:5])
            if character.upper() in 'ABCDEFGHIJKLMNOPQRSTUWXYZŃÓŚĆŹŻŁĄĘ':
                self.boxes.append([character.upper(),left,bottom,right,top])
            else:
                self.boxes.append(['?',left,bottom,right,top])
        #Transforming current structure into rows of letters. Letter's row is determined based on coordinates of its left border.
        i = 0
        row=[self.boxes[0]]
        while True:
            i+=1
            if i >= len(self.boxes):
                self.letters.append(row)
                break
            if self.boxes[i][1]> self.boxes[i-1][1]:
                row.append(self.boxes[i])
                continue
            self.letters.append(row)
            row=[self.boxes[i]]

        #Equalising length of all rows based on average length.
        average = round(sum(map(len,self.letters))/len(self.letters))
        if not all(x == average for x in map(len,self.letters)):
            for row in self.letters:
                length=len(row)
                if length == average:
                    continue
                if abs(length-average)>=2:
                    raise RuntimeError('OCR has messed up or the image contains too much noise')
                if length<average:
                    row.append("?")
                    continue
                if '?' in row:
                    del row[row.index('?')]
                    continue
                del row[-1]                
        
    #SENS - sensitivity of denoising the image and later dialation (recommended values range:0-5), threshold- value that determines the brightness of a pixel needed to round it to 255 (recommended values range:130-170)
    def noise_reduciton(self, binary, SENS:int, file_path='') -> cv2.typing.MatLike:
        
        #creating a kernel and using it do denoise our image and remove grid lines
        binary=cv2.bitwise_not(binary)
        horizontal_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        vertical_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        sqaure_kernel = np.ones((SENS,SENS),np.uint8)

        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_line_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_line_kernel, iterations=2)
        all_detected_lines = cv2.bitwise_and(cv2.bitwise_not(horizontal_lines),cv2.bitwise_not(vertical_lines))

        gridless = cv2.bitwise_and(binary, binary, mask=all_detected_lines)
        denoised = cv2.morphologyEx(gridless,cv2.MORPH_OPEN, sqaure_kernel)
        dialated_letters = cv2.dilate(denoised,sqaure_kernel)

        processed_image = cv2.bitwise_not(dialated_letters)

        if file_path!='':
            result = Image.fromarray((processed_image).astype(np.uint8))
            result.save(file_path)
        return processed_image

class CrosswordSolver:
    def __init__(self ,words, letters,MIN_WORD_LENGTH=2):
        self.words = words
        self.WIDTH = len(letters[0])
        self.HEIGTH = len(letters)
        self.letters = [letter for row in letters for letter in row]
        self.used_letters_index = set()
        self.MIN_WORD_LENGTH=MIN_WORD_LENGTH
    
    #Counting the amount of differences in 2 strings.
    def count_differences(self,string1,string2) ->  int:
        return sum(1 for a,b in zip(string1,string2) if a!=b)
    
    #Solving the puzzle with varying levels of uncertanty allowed.
    def solve(self, tolerance=1)-> None:
        for current_tolarance in range(tolerance+1):
            #Horizontal check 
            i=0
            #Looping through all rows and checking for possible words in them 
            while i<self.HEIGTH:
                start=i*self.WIDTH
                stop=start+self.MIN_WORD_LENGTH
                while stop<=(i+1)*self.WIDTH:
                    n=0
                    while n<len(self.words):                 
                        word=self.words[n]
                        current_word=''.join(self.letters[start:stop])
                        if self.count_differences(current_word,word)<=current_tolarance or self.count_differences(current_word,word[::-1])<=current_tolarance:
                            if len(word)!=len(current_word):
                                if start+len(word)>self.WIDTH*(i+1):
                                    n+=1
                                    continue
                                stop=start+len(word)
                                continue
                            del self.words[n]
                            self.used_letters_index.update([current_index for current_index in range(start,stop)])
                        n+=1
                    if stop-start>self.MIN_WORD_LENGTH-1:
                        start+=1
                        continue
                    stop+=1
                i+=1

            #vertical
            #Looping through all columns and checking for possible words in them 
            i = 0 
            while i < self.WIDTH:
                start = i 
                stop = start+self.WIDTH*self.MIN_WORD_LENGTH
                while stop<len(self.letters)+i:
                    n=0 
                    while n<len(self.words):
                        word=self.words[n]
                        current_word=''.join(self.letters[index] for index in range(start,stop,self.WIDTH))
                        if self.count_differences(current_word,word)<=current_tolarance or self.count_differences(current_word,word[::-1])<=current_tolarance:
                            if len(word)!=len(current_word):
                                if start+len(word)*self.WIDTH>len(self.letters)+i:
                                    n+=1
                                    continue
                                stop=start+len(word)*self.WIDTH
                                continue
                            del self.words[n]
                            self.used_letters_index.update([current_index for current_index in range(start,stop,self.WIDTH)])
                        n+=1
                    if (stop-start)/self.WIDTH>self.MIN_WORD_LENGTH-1:
                        start+=self.WIDTH
                        continue
                    stop+=self.WIDTH
                i+=1
            #diagonal right
            #Looping through all diagonals on left and top side and checking for possible words in them 
            i = 0
            while i<self.WIDTH+self.HEIGTH-(self.MIN_WORD_LENGTH-1)*2-1:
                
                start = self.WIDTH*(self.HEIGTH-self.MIN_WORD_LENGTH)-i*self.WIDTH
                if start<0:
                    start=int(start/-self.WIDTH)
                stop = start+(self.WIDTH+1)*self.MIN_WORD_LENGTH
                while stop<len(self.letters)+self.WIDTH and stop%self.WIDTH>max(i-self.HEIGTH+self.MIN_WORD_LENGTH,0)%self.WIDTH:
                    n=0 
                    while n<len(self.words):
                        word=self.words[n]
                        current_word=''.join(self.letters[index] for index in range(start,stop,self.WIDTH+1))

                        if self.count_differences(current_word,word)<=current_tolarance or self.count_differences(current_word,word[::-1])<=current_tolarance:
                            if len(word)!=len(current_word):
                                if start+len(word)*(self.WIDTH+1)>len(self.letters)+self.WIDTH:
                                    n+=1
                                    continue
                                stop=start+len(word)*(self.WIDTH+1)
                                continue
                            del self.words[n]
                            self.used_letters_index.update([current_index for current_index in range(start,stop,self.WIDTH+1)])
                        n+=1
                    if (stop-start)/(self.WIDTH+1)>self.MIN_WORD_LENGTH-1:
                        start+=self.WIDTH+1
                        continue
                    stop+=self.WIDTH+1
                i+=1
            #diagonal left
            i = 0
            #Looping through all diagonals on right and top side and checking for possible words in them 
            while i<self.WIDTH+self.HEIGTH-(self.MIN_WORD_LENGTH-1)*2-1:
                start = self.WIDTH*(self.HEIGTH-self.MIN_WORD_LENGTH+1)-i*self.WIDTH-1
                if start<=0:
                    start=self.WIDTH-int(start/-self.WIDTH)-2
                stop = start+(self.WIDTH-1)*self.MIN_WORD_LENGTH
                while stop<len(self.letters)+self.WIDTH-i and stop%(self.WIDTH-1)<self.WIDTH-max(i-self.HEIGTH+self.MIN_WORD_LENGTH,0)%self.WIDTH:
                    n=0 
                    while n<len(self.words):
                        word=self.words[n]
                        current_word=''.join(self.letters[index] for index in range(start,stop,self.WIDTH-1))
                        if self.count_differences(current_word,word)<=current_tolarance or self.count_differences(current_word,word[::-1])<=current_tolarance:
                            if len(word)!=len(current_word):
                                if start+len(word)*(self.WIDTH-1)>len(self.letters)+self.WIDTH:
                                    n+=1
                                    continue
                                stop=start+len(word)*(self.WIDTH-1)
                                continue
                            del self.words[n]
                            self.used_letters_index.update([current_index for current_index in range(start,stop,self.WIDTH-1)])
                        n+=1
                    if (stop-start)/(self.WIDTH-1)>self.MIN_WORD_LENGTH-1:
                        start+=self.WIDTH-1
                        continue
                    stop+=self.WIDTH-1
                i+=1
    def anwser_after_solving(self)->str:
        output=''
        output_list=[]
        for number in range(self.HEIGTH*self.WIDTH):
            if number not in self.used_letters_index:
                output+=self.letters[number]
                output_list.append(number)
        return output,output_list
    


#Example usage
samples_file_path = r"E:/Programowanie/pajton/solver/samples/sample8/"
try:
    letters_image= cv2.imread(samples_file_path+'/letters.png')
    words_image= cv2.imread(samples_file_path+'/words.png')

    converter= ImageConverter(letters=letters_image, words=words_image)
    #Threshold and SENS need to be adjusted based on picked sample (threshold between 130-180 and SENS between 0-5)
    converter.letters_image_OCR(preprocessing_threshold=150, SENS=4)
    converter.words_image_OCR(preprocessing_threshold=140)

    solver = CrosswordSolver(converter.words,[[value[0] for value in row] for row in converter.letters], MIN_WORD_LENGTH=5)
    solver.solve(tolerance=1,)
    anwser,indexes=solver.anwser_after_solving()
    print(f'The anwser is: "{anwser}" with each letters index: {indexes}.')
except Exception as e:
    print(e)
