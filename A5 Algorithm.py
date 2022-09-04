from email import message
import pyaudio
import wave
import threading
import numpy as np
import random
from textwrap import wrap
from scipy.io.wavfile import write,read
from tkinter import *
import time
from tkinter.ttk import *
import string

general_window = None           # represents the main GUI window.
label_encrypting_now = None     # Label indcating that the encryption is happening now.
pb1 = None                      # Progress bar to indicate that recording is in process.
label_try_again = None          # Label indcating that the user can try again the recording process.
execution_option_button = None  # Button that trigers the record operation execution
file_name = None                # The prefix of encrypted/decrypted wav file.

class A5Register:
    """
        This class represents an LFSR register in the A5/1 Algorithm.
    """
    def __init__(self, id, length, clockingBit, tappedBits):
        """Constructor of the A5Register Class
        Constructor of the registers that can be used in A5/1 Algorithm.
        Args:
            id (int): Represents the Id of the register.
            length (int): Represents the length of the register.
            clockingBit (int): The index of the clocking bit.
            tappedBits (list of int): The list of indexes of the tapped bits.
        """
        self.id = id
        self.length = length
        self.register = [0 for _ in range(length)]
        self.clockingBit = clockingBit
        self.tappedBits = tappedBits

    def __str__(self):
        """String representation

        Returns:
            str: Register Object details.
        """
        message = f"Object ID = {self.getId()}\nLSRF = {self.getLFSRRegister()}\n"
        message += f"Tapped bits indexes = {self.getIndexesTappedBits()}\n"
        message += f"Clocking bit value = {self.getValueClockingBit()}\n"
        message += f"Clocking bit index = {self.getIndexClockingBit()}\n"
        return message

    def __repr__(self):
        """Object representation

        Returns:
            str: Register Object details.
        """
        message = f"Object ID = {self.getId()}\nLSRF = {self.getLFSRRegister()}\n"
        message += f"Tapped bits indexes = {self.getIndexesTappedBits()}\n"
        message += f"Clocking bit value = {self.getValueClockingBit()}\n"
        message += f"Clocking bit index = {self.getIndexClockingBit()}\n"
        return message

    # Setters
    def setId(self, id):
        """setId
        Setter of the ID of the register.
        Args:
            id (int): The ID of the register.
        """					
        self.id = id

    def setLength(self, length):
        """setLength
        Setter of the length of the register.
        Args:
            length (int): The length of the register.
        """		
        self.length = length

    def setLFSRRegister(self, register):
        """setLFSRRegister
        Setter of the register.
        Args:
            register (list of int):  Register that is a list of int.
        """	        
        self.register = register
    def setIndexClockingBit(self, clockingBit): 	
        """setClockingBit

        Args:
            clockingBit (int): The index of the clocking bit.
        """
        self.clockingBit = clockingBit
    def setIndexesTappedBits(self, tappedBits): 
        """setIndexesTappedBits
        Setter of the ID of the register.
        Args:
            tappedBits (list of int): The list of indexes of the tapped bits.
        """		
        self.tappedBits = tappedBits

    # Getters
    def getId(self):
        """getId
        Getter of the ID of the register.
        Returns:
            int: The ID of the register.
        """
        return self.id

    def getLFSRRegister(self):
        """getLFSRRegister
        Getter of the register itself.
        Returns:
            list of int: Getter of the list of integers that represents the LFSR Register.
        """
        return self.register

    def getBitByIndex(self, index):
        """getBitByIndex

        Returns:
            bit(zero or one): Getter of the bit value in a specific index.
        """			    
        return self.register[index]

    def getBits(self, indexesList):
        """_summary_
        Get the values of bits in specific indexes.
        Args:
            indexesList (list of int): List of indexes of bits.

        Returns:
            list of int: Tuple of values of bits.
        """
        return [self.getBitByIndex(index) for index in indexesList]

    def getRightMostBit(self):
        """_summary_
        Get the last bit - right most bit.
        Returns:
            int: Right most bit.
        """
        return self.register[-1]

    def getLength(self):
        """getLength

        Returns:
            int: Getter of the length of the register(19,22 or 23).
        """
        return self.length

    def getIndexClockingBit(self):
        """getIndexClockingBit

        Returns:
            int: Getting the index of the clocking bit
        """
        return self.clockingBit

    def getValueClockingBit(self):
        """getClockingBit

        Returns:
            int: Getting the value of the clocking bit
        """
        return self.register[self.clockingBit]

    def getIndexesTappedBits(self):
        """getTappedBits

        Returns:
            list of int: Getter of the indexes of tapped bits.
        """
        return self.tappedBits
    
    def getValuesTappedBits(self):
        """
        Get values of the tapped bits.
        Returns:
            list of int: Tapped bits values.
        """
        return self.getBits(self.getIndexesTappedBits())

    def getXORValuesTappedBits(self, bit=0):
        """
        Get values of the tapped bits.
        Returns:
            list of int: Tapped bits values.
        """
        result = bit
        for bitIndex in self.getIndexesTappedBits():
            result ^= self.register[bitIndex]
        return result

def initializeLFSRRegisters(): 
    """Phase 1
    Initializing the three registers of A5/1 Algorithm with zeros.
    Returns:
        (A5Register, A5Register, A5Register) : The A5 Registers initialized with zeros.
    """
    lfsr1 = A5Register(1,19,8,[13,16,17,18])
    lfsr2 = A5Register(2,22,10,[20,21])
    lfsr3 = A5Register(3,23,10,[7,20,21,22])
    return (lfsr1, lfsr2, lfsr3)

def getMajority(*bits):
    """getMajority
    Get the majority of passed bits.
    Returns:
        int: The result of Majority.
    """
    numberOfOnes = sum([int(bit) for bit in bits])
    if(numberOfOnes > 1):
        return 1
    return 0

def executeBitsXOR(*bits):
    """executeBitsXOR
    Execute xor on the passed parameters.
    Returns:
        int: The result of XOR.
    """
    numberOfOnes = sum([int(bit) for bit in bits[0]])
    if(numberOfOnes % 2 == 0):
        return 0
    return 1

def xor(*bits):
    """xor
    Execute xor on the passed 2 parameters.
    Returns:
        int: The result of XOR.
    """
    numberOfOnes = sum([int(bit) for bit in bits])
    if(numberOfOnes % 2 == 0):
        return 0
    return 1


globalSessionKey = None
def generateSessionKey():
    """Generation of the session key that consist of 64 bits

    Returns:
        string: session key of 64 bits.
    """
    global globalSessionKey
    if not globalSessionKey:
        globalSessionKey = ''.join([str(random.randint(0,1)) for _ in range(64)])
    return globalSessionKey

def clockWithSessionKey(lfsr1,lfsr2,lfsr3,sessionKey):
    """The second phase of the A5/1 Algorithm

    Args:
        lfsr1 (list of int): First Register of the A5/1 Algorithm.
        lfsr2 (list of int): Second Register of the A5/1 Algorithm.
        lfsr3 (list of int): Third Register of the A5/1 Algorithm.
        sessionKey (string): session key of 64 bits.
    """
    for bit in sessionKey:
        bit = int(bit)
        leftMostBit1 = lfsr1.getXORValuesTappedBits(bit)
        leftMostBit2 = lfsr2.getXORValuesTappedBits(bit)
        leftMostBit3 = lfsr3.getXORValuesTappedBits(bit)
        lfsr1.setLFSRRegister([leftMostBit1]+lfsr1.getLFSRRegister()[0:lfsr1.getLength()-1])
        lfsr2.setLFSRRegister([leftMostBit2]+lfsr2.getLFSRRegister()[0:lfsr2.getLength()-1])
        lfsr3.setLFSRRegister([leftMostBit3]+lfsr3.getLFSRRegister()[0:lfsr3.getLength()-1])

def generateCounter(frameCounter):
    """_summary_
    Generate a Counter of 2 bits to count the frames.
    Args:
        frameCounter (string): String represents a string of bits.

    Returns:
        string : The modified string of bits - expanded to be 22 bits.
    """
    return '0'*(22-len(frameCounter)) + frameCounter

def clockWithFrameCounter(lfsr1,lfsr2,lfsr3,frameCounter):
    """The third phase of the A5/1 Algorithm

    Args:
        lfsr1 (list of int): First Register of the A5/1 Algorithm.
        lfsr2 (list of int): Second Register of the A5/1 Algorithm.
        lfsr3 (list of int): Third Register of the A5/1 Algorithm.
        frameCounter (string of 22 bits): Indictaes the order of the frame that is needed to be encrypted.
    """
    for bit in frameCounter:
        bit = int(bit)
        leftMostBit1 = lfsr1.getXORValuesTappedBits(bit)
        leftMostBit2 = lfsr2.getXORValuesTappedBits(bit)
        leftMostBit3 = lfsr3.getXORValuesTappedBits(bit)
        lfsr1.setLFSRRegister([leftMostBit1]+lfsr1.getLFSRRegister()[0:lfsr1.getLength()-1])
        lfsr2.setLFSRRegister([leftMostBit2]+lfsr2.getLFSRRegister()[0:lfsr2.getLength()-1])
        lfsr3.setLFSRRegister([leftMostBit3]+lfsr3.getLFSRRegister()[0:lfsr3.getLength()-1])

def clockIrregualrClocking100(lfsr1,lfsr2,lfsr3):
    """Phase Four
    Clock Registers with irregualr clocking.
    Args:
        lfsr1 (A5Register): The first register of A5/1 Algorithm.
        lfsr2 (A5Register): The second register of A5/1 Algorithm.
        lfsr3 (A5Register): The third register of A5/1 Algorithm.
    """
    for _ in range(100):
        clockingBit1 = lfsr1.getValueClockingBit()
        clockingBit2 = lfsr2.getValueClockingBit()
        clockingBit3 = lfsr3.getValueClockingBit()
        majorityBit = getMajority(clockingBit1, clockingBit2, clockingBit3)
        if(clockingBit1 == majorityBit):
            leftMostBit1 = lfsr1.getXORValuesTappedBits()
            lfsr1.setLFSRRegister([leftMostBit1]+lfsr1.getLFSRRegister()[0:lfsr1.getLength()-1])
        if(clockingBit2 == majorityBit):
            leftMostBit2 = lfsr2.getXORValuesTappedBits()
            lfsr2.setLFSRRegister([leftMostBit2]+lfsr2.getLFSRRegister()[0:lfsr2.getLength()-1])
        if(clockingBit3 == majorityBit):
            leftMostBit3 = lfsr3.getXORValuesTappedBits()
            lfsr3.setLFSRRegister([leftMostBit3]+lfsr3.getLFSRRegister()[0:lfsr3.getLength()-1])

def produceKeyStream(lfsr1, lfsr2, lfsr3):
    """Phase 5
    Produce a Key Stream.
    Args:
        lfsr1 (A5Register): The first register of A5/1 Algorithm.
        lfsr2 (A5Register): The second register of A5/1 Algorithm.
        lfsr3 (A5Register): The third register of A5/1 Algorithm.

    Returns:
        string: A string that represents the key stream.
    """
    keyStream = []
    for _ in range(228):
        keyStream.append(str(executeBitsXOR([lfsr1.getRightMostBit(),lfsr2.getRightMostBit(), lfsr3.getRightMostBit()])))
        clockingBit1 = lfsr1.getValueClockingBit()
        clockingBit2 = lfsr2.getValueClockingBit()
        clockingBit3 = lfsr3.getValueClockingBit()
        majorityBit = getMajority(clockingBit1, clockingBit2, clockingBit3)
        if(clockingBit1 == majorityBit):
            leftMostBit1 = lfsr1.getXORValuesTappedBits()
            lfsr1.setLFSRRegister([leftMostBit1]+lfsr1.getLFSRRegister()[0:lfsr1.getLength()-1])
        if(clockingBit2 == majorityBit):
            leftMostBit2 = lfsr2.getXORValuesTappedBits()
            lfsr2.setLFSRRegister([leftMostBit2]+lfsr2.getLFSRRegister()[0:lfsr2.getLength()-1])
        if(clockingBit3 == majorityBit):
            leftMostBit3 = lfsr3.getXORValuesTappedBits()
            lfsr3.setLFSRRegister([leftMostBit3]+lfsr3.getLFSRRegister()[0:lfsr3.getLength()-1])
    return ''.join(keyStream)

def encryptFrame(plaineText, KeyStream):
    """encryptFrame
    Encryption of one frame in A5/1 Algorithm.
    Args:
        plaineText (string): string of bits.
        KeyStream (string): string of bits.

    Returns:
        string: String of bits.
    """
    cipher = "".join([str(xor(plaineText[index], KeyStream[index])) for index in range(len(KeyStream))])
    return cipher

def convertStringToFrames(plaineText):
    """Conversion of long string to frame each of 228 bits.

    Args:
        plaineText (string): The plaine Text message.

    Returns:
        list of strings: Frames that represent the plain message.
    """
    listOfFrames = [plaineText[i:i + 228] for i in range(0, len(plaineText), 228)]
    lastFrame = listOfFrames[-1]
    while(len(lastFrame) % 228 != 0):
        lastFrame += "0"
    listOfFrames[-1] = lastFrame
    return listOfFrames

def encryptMessage(plaineText, sessionKey, frameCounter=0):
    """_summary_
    Encryption of the whole plain text.
    Args:
        plaineText (string): The plain text.
        sessionKey (string): The session key.
        frameCounter (int, optional): The frame counter. Defaults to 0.

    Returns:
        string: The cipher text
    """
    cipher = ""
    counter = frameCounter
    listOfFrames = convertStringToFrames(plaineText)
    for frame in listOfFrames:
        binaryFrameCounter = generateCounter(bin(counter)[2:])
        lfsr1, lfsr2, lfsr3 = initializeLFSRRegisters()
        clockWithSessionKey(lfsr1, lfsr2, lfsr3, sessionKey)
        clockWithFrameCounter(lfsr1, lfsr2, lfsr3, binaryFrameCounter)
        clockIrregualrClocking100(lfsr1, lfsr2, lfsr3)
        keyStream = produceKeyStream(lfsr1, lfsr2, lfsr3)
        cipher += encryptFrame(frame, keyStream)
        counter = counter + 1
    return cipher

def record():
    """record
    Function to perform the recording with encryption and then decryption.
    """
    global execution_option_button
    global label_try_again
    global label_encrypting_now
    global file_name
    label_encrypting_now = Label(general_window, text = "Encrypting Now...")
    if(label_try_again):
        label_try_again.destroy()
    label_encrypting_now.place(x = 70,y = 90) 
    file_name = ''.join([random.choice(string.ascii_uppercase + string.digits) for _ in range(5)])
    chunk = 1024 
    sample_format = pyaudio.paInt16 
    channels = 2
    fs = 44100
    seconds = 1
    p = pyaudio.PyAudio() 
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    audio_bits = ''
    frames = []
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
        l = [bin(x)[2:].zfill(16) for x in np.frombuffer(data, dtype=np.uint16)]
        audio_bits += ''.join(l)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    frames = b''.join(frames)
    wf=wave.open(file_name+'.wav', 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(frames)
    wf.close()
    encrypted_bits = encryptMessage(audio_bits, generateSessionKey())
    result = [encrypted_bits[i:i + 16] for i in range(0, len(encrypted_bits), 16)]
    result = [int(x, 2) for x in result]
    result = np.int16(result)

    wf=wave.open(file_name+'_encrypted.wav', 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(result.tobytes())
    wf.close()

    decrpyt(result)

    execution_option_button['state'] = NORMAL
    label_encrypting_now.destroy()
    label_try_again = Label(general_window, text = "You Can Try Again")
    label_try_again.place(x = 70,y = 90)


def decrpyt(data: np.ndarray):
    """decyrpy
    Decryption part of A5/1 algorithm.
    Args:
        data (np.ndarray): numpy array of int elements.
    """
    global file_name
    global label_encrypting_now
    label_encrypting_now.destroy()
    label_encrypting_now = Label(general_window, text = "Decrypting Now...")
    label_encrypting_now.place(x = 70,y = 90) 
    data = data.astype(np.uint16)
    data = ''.join([bin(x)[2:].zfill(16) for x in data])
    decrypted_bits = encryptMessage(data, generateSessionKey())
    result = [decrypted_bits[i:i + 16] for i in range(0, len(decrypted_bits), 16)]
    result = [int(x, 2) for x in result]
    result = np.int16(result)

    channels = 2
    fs = 44100
    sample_format = pyaudio.paInt16
    p = pyaudio.PyAudio()
    wf = wave.open(file_name + '_decrypted.wav', 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(result.tobytes())
    wf.close()


def recordThread():
    """recordThread
    A thread to handle the recording operation.
    """
    global execution_option_button
    execution_option_button['state'] = DISABLED
    threading.Thread(target= record).start()

def step():
    """step
    Function to handle the progress of the Progress Bar.
    """
    global general_window
    global pb1
    global label_encrypting_now
    global text
    pb1 = Progressbar(general_window, orient=HORIZONTAL, length=130, mode='indeterminate')
    pb1.pack(expand=True)
    pb1.place(x = 70, y = 90)
    for _ in range(10):
        general_window.update_idletasks()
        pb1['value'] += 10
        time.sleep(0.1)
    pb1.destroy()
    general_window.update_idletasks()

def gui():
    """gui
    The main GUI window of the application.
    """
    global execution_option_button
    global general_window
    global pb1
    global label_encrypting_now
    global label_try_again
    general_window = Tk()
    general_window.geometry("425x150")
    general_window.title("A5 Algorithm")
    style = Style()
    style.configure('TButton', font = ('Arial', 8, 'bold'), borderwidth = '2')
    style.map('TButton', foreground = [('active', '!disabled', 'green')],
                        background = [('active', 'black')])
    choose_operation_label = Label(general_window,text = "Please choose the operation you want to perform: ")
    choose_operation_label.place(x = 0 , y = 5)
    execution_option_button = Button(general_window,text = 'Record and (En/De)crypt',command = lambda:[step(),recordThread()])
    execution_option_button.place(x = 70 , y = 40)
    general_window.mainloop()

# gui()

sessionKey = "0100111000101111010011010111110000011110101110001000101100111010"
frameCounter = "1110101011001111001011"
print("Session key length : ", len(sessionKey))
print("Frame Counter : ", len(frameCounter))
lfsr1, lfsr2, lfsr3 = initializeLFSRRegisters()
print("Step of initialization")
print(lfsr1, lfsr2, lfsr3)
clockWithSessionKey(lfsr1, lfsr2, lfsr3, sessionKey)
print("Step of clocking with session key 64")
print(lfsr1, lfsr2, lfsr3)
print("-----------------------------------------------")
# lfsr1.setLFSRRegister("1 0 1 0 1 1 0 1 0 0 1 1 1 1 0 0 1 1 0".split(" "))
# lfsr2.setLFSRRegister("0 1 0 0 1 0 0 1 0 1 0 0 1 1 1 1 1 0 0 0 1 1".split(" "))
# lfsr3.setLFSRRegister("0 1 0 0 1 0 0 0 0 1 0 0 1 1 0 0 0 1 0 1 0 1 1".split(" "))
clockWithFrameCounter(lfsr1, lfsr2, lfsr3, frameCounter)
print("Clock with Frame Counter 22")
print(lfsr1, lfsr2, lfsr3)
print("-----------------------------------------------")
clockIrregualrClocking100(lfsr1, lfsr2, lfsr3)
print("Step of clocking 100 times with Irregualr clock")
print(lfsr1, lfsr2, lfsr3)
print("-----------------------------------------------")
keyStream = produceKeyStream(lfsr1, lfsr2, lfsr3)
print("step of Producing the key stream")
print("-------------------------------------------------")
print("The key stream")
print(keyStream)
print("The registers are :")
print(lfsr1, lfsr2, lfsr3)
# print("-----------------------------------------------")
# plaineText = bin(pow(2,240))[2:][0:228]
plaineText = bin(pow(5,1))[2:]
# plaineText = bin(pow(2,227))[2:]
print(len(plaineText))
frames = convertStringToFrames(plaineText)
cipher = ""
for frame in frames:
    cipher += encryptFrame(frame, keyStream)
print("--------")
print("The first and last five bits of plain  text : \n", frame)
print("The first and last five bits of key  stream : \n", keyStream)
print("The first and last five bits of cipher text : \n", cipher)