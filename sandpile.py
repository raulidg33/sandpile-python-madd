import os
import cv2
import time
import imageio
import datetime
import numpy as np
import matplotlib.pyplot as plt

from typing import Literal

class Sandpile:
    # ==================================================================================================================
    # INITIALIZATION FUNCTION
    # ==================================================================================================================
    def __init__(self, grid:np.ndarray=None, gdim0:int=13, crit:int=4):
        gdim0 = gdim0 if gdim0%2 != 0 else gdim0+1

        self.__gdim0 = gdim0
        self.__grid = grid if grid else np.zeros((gdim0, gdim0))
        self.__frames = list()
        self.__frames.append(self.grid.copy())
        self.__center = gdim0//2
        self.__crit = crit
        self.__colormap = 0
        self.__times = []
        img = cv2.applyColorMap((self.__grid*255/self.__crit).astype(np.uint8), 0)
        self.__img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ==================================================================================================================
    # PROPERTIES
    # ==================================================================================================================
    @property
    def grid(self):
        return self.__grid
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def size(self):
        return self.grid.shape[0]
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def crit(self):
        return self.__crit
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def img(self):
        return self.__img
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def colormap(self):
        return self.__colormap
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def times(self):
        return self.__times
    # ==================================================================================================================
    # PRIVATE FUNCTIONS
    # ==================================================================================================================
    def __choose_pile(self):
        crits = np.argwhere(self.grid>self.crit)
        return crits[np.random.choice(len(crits))]
    # ------------------------------------------------------------------------------------------------------------------
    def __check_and_fix_size(self, x:int, y:int):
        if (x+1 >= self.size) or (x-1 < 0) or (y+1 >= self.size) or (y-1 < 0):
            self.pad_grid()
            return True
        return False
    # ------------------------------------------------------------------------------------------------------------------
    def __update_frames(self):
        nframe = self.grid.copy()
        self.__frames.append(nframe)
    # ==================================================================================================================
    # GETTERS AND SETTERS
    # ==================================================================================================================
    def get_pile(self, x:int, y:int) -> int:
        return self.grid[x,y]
    # ------------------------------------------------------------------------------------------------------------------
    def set_pile(self, x:int, y:int, grains:int):
        self.__grid[x, y] = grains
    # ------------------------------------------------------------------------------------------------------------------
    def set_crit(self, new_crit:int):
        self.__crit = new_crit
    # ------------------------------------------------------------------------------------------------------------------
    def set_colormap(self, colormap:int):
        self.__colormap = colormap
    # ------------------------------------------------------------------------------------------------------------------
    def clear(self, clear_times=False):
        self.__grid = np.zeros(shape=(self.__gdim0, self.__gdim0))
        self.__center = self.__gdim0//2
        if clear_times:
            self.__times = list()
    # ==================================================================================================================
    # FUNCTIONS TO RUN THE MODEL
    # ==================================================================================================================
    def increase_pile(self, x:int, y:int, grains:int):
        self.__grid[x, y] += grains
    # ------------------------------------------------------------------------------------------------------------------
    def pad_grid(self):
        self.__grid = np.pad(self.grid, (1,1))
        self.__center = self.grid.shape[0]//2
    # ------------------------------------------------------------------------------------------------------------------
    def collapse_normal(self, x:int, y:int):
        cmod = self.grid[x,y]%4
        cdiv = self.grid[x,y]//4

        resized = self.__check_and_fix_size(x, y)
        if resized: 
            x+=1
            y+=1

        self.set_pile(x, y, cmod)
        self.increase_pile(x+1, y, cdiv)
        self.increase_pile(x-1, y, cdiv)
        self.increase_pile(x, y+1, cdiv)
        self.increase_pile(x, y-1, cdiv)
    # ------------------------------------------------------------------------------------------------------------------
    def collapse_diagonal(self, x:int, y:int):
        cmod = self.grid[x,y]%4
        cdiv = self.grid[x,y]//4

        resized = self.__check_and_fix_size(x, y)
        if resized: 
            x+=1
            y+=1

        self.set_pile(x, y, cmod)
        self.increase_pile(x+1, y+1, cdiv)
        self.increase_pile(x+1, y-1, cdiv)
        self.increase_pile(x-1, y+1, cdiv)
        self.increase_pile(x-1, y-1, cdiv)
    # ------------------------------------------------------------------------------------------------------------------
    def collapse_full(self, x:int, y:int):
        cmod = self.grid[x,y]%8
        cdiv = self.grid[x,y]//8

        resized = self.__check_and_fix_size(x, y)
        if resized: 
            x+=1
            y+=1
    
        self.set_pile(x, y, cmod)
        self.increase_pile(x+1, y, cdiv)
        self.increase_pile(x-1, y, cdiv)
        self.increase_pile(x, y+1, cdiv)
        self.increase_pile(x, y-1, cdiv)
        self.increase_pile(x+1, y+1, cdiv)
        self.increase_pile(x+1, y-1, cdiv)
        self.increase_pile(x-1, y+1, cdiv)
        self.increase_pile(x-1, y-1, cdiv)
    # ------------------------------------------------------------------------------------------------------------------
    def collapse_random(self, x:int, y:int):
        choice = np.random.randint(3)
        if choice == 0:
            self.collapse_normal(x, y)
        elif choice == 1:
            self.collapse_diagonal(x, y)
        else:
            self.collapse_full(x, y)
    # ==================================================================================================================
    # THE RUN FUNCTION
    # ==================================================================================================================
    def run(self, 
            n:int, 
            start:tuple[int,int]=None, 
            collapse:Literal['normal', 'diagonal', 'full', 'random']='normal',
            save_frames:bool=True):
        if start==None:
            self.set_pile(self.__center, self.__center, n)
        else:
            self.set_pile(start[0], start[1], n)
        if save_frames:
            self.__update_frames()

        if collapse not in ['normal', 'diagonal', 'full', 'random']:
            raise ValueError(f'collapse value "{collapse}" not valid. Valid options are normal, diagonal, full or random.')
        
        if collapse == 'full' and self.crit<8:
            raise ValueError('collapse value "full" requires crit value to be >=8. You can change it with set_crit() function.')
        start_time = time.time()
        while (self.grid > self.crit).any():
            choice = self.__choose_pile()

            if collapse == 'normal':
                self.collapse_normal(choice[0], choice[1])
            elif collapse == 'diagonal':
                self.collapse_diagonal(choice[0], choice[1])
            elif collapse == 'full':
                self.collapse_full(choice[0], choice[1])
            elif collapse == 'random':
                self.collapse_random(choice[0], choice[1])
            if save_frames:
                self.__update_frames()

        self.change_colormap()
        self.__times.append(time.time() - start_time)
    # ==================================================================================================================
    # FUNCTIONS TO VIEW OR TO ANIMATE
    # ==================================================================================================================
    def change_colormap(self, colormap: int=0):
        img = cv2.applyColorMap((self.grid*255/self.crit).astype(np.uint8), colormap)
        self.__img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ------------------------------------------------------------------------------------------------------------------
    def show_img(self):
        plt.imshow(self.img)
        plt.axis('off')
        plt.show()
    # ------------------------------------------------------------------------------------------------------------------
    def __check_and_create_output(self):
        path = os.path.join(os.getcwd(), 'output/')
        if not os.path.exists(path):
            os.mkdir(path)
        return path
    # ------------------------------------------------------------------------------------------------------------------
    def save_as_gift(self, path:str=None, duration=0.02, resize=False):
        frames = self.__frames
        if path == None:
            path = self.__check_and_create_output()
        filename = os.path.join(path, f'{datetime.datetime.timestamp(datetime.datetime.now())}.gif')

        ndim = self.size*40 # new dimension
        ndim += ndim%16 # adding extra pixels for compatibility with most codecs and players

        writer = imageio.get_writer(filename, duration=duration, mode='I')
        for frame in frames:
            padw = (self.size - frame.shape[0])//2
            frame = np.pad(frame, (padw, padw))
            frame = cv2.cvtColor(cv2.applyColorMap((frame*255/self.crit).astype(np.uint8), self.colormap), cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, (ndim,ndim), interpolation = cv2.INTER_AREA)

            writer.append_data(frame)
        writer.close()
    # ------------------------------------------------------------------------------------------------------------------
    def save_as_mp4(self, path:str=None, fps=120, resize=False):
        frames = self.__frames
        if path == None:
            path = self.__check_and_create_output()
        filename = os.path.join(path, f'{datetime.datetime.timestamp(datetime.datetime.now())}.mp4')

        writer = imageio.get_writer(filename, fps=fps)

        ndim = self.size*40
        ndim += ndim%16

        for frame in frames:
            padw = (self.size - frame.shape[0])//2
            frame = np.pad(frame, (padw, padw))
            frame = cv2.cvtColor(cv2.applyColorMap((frame*255/self.crit).astype(np.uint8), self.colormap), cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, (ndim,ndim), interpolation = cv2.INTER_AREA)
            
            writer.append_data(frame)
        writer.close()