"""Set of video utilities to make dealing with frames in a video easier."""
import os

class VideoObj():
    """Class to work with video instances. 
    This class stores information about videos and provides utility functions to navigate through the video."""
    def __init__(self, images, fps=30):
        self.images = images
        self.fps = fps
        self.length = len(images)
        self.caption_memory = dict()
        self.vqa_memory = dict()
        self.explanations = list()
    
    def __len__(self):
        """Returns the length of the video in frames."""
        return self.length

    def __getitem__(self, index):
        """Returns the frame at the given index."""
        return self.images[index]

    def get_second(self, second):
        """Retrieve a specific frame at a given second"""
        idx = int(second * self.fps)
        return [idx], self.images[idx]
     
    def convert_to_frame(self, second):
        """Helper function to convert a second to a frame index"""
        return int(second * self.fps)
        
    def select_seconds_range(self, start, end):
        """Get a range of frames from the video. 
        Enforces that the start and end are within the video length.
        """
        start_idx = min(0, int(start * self.fps))
        end_idx = min(int(end * self.fps), self.length)    
        return self.images[start_idx, end_idx]

    def get_block(self, second, length=1, method="center"):
        """Get a range of frames from the video.
            - length: length of the block in seconds
            - method: method to select the block. 
                "center" selects the block around the given second, 
                "start" selects the block starting at the given second, 
                "end" selects the block ending at the given second.
        """
        if method == "center":
            start = second - length // 2, 0
            end = second + length // 2
        elif method == "start":
            start = second
            end = second + length
        elif method == "end":
            start = second - length
            end = second
        return self.select_seconds_range(start, end)

    def move(self, current, direction, step_size=1):
        """Move forward or backward in the video by retrieving the corresponding block `step_size` seconds away."""
        assert direction in ["forward", "backward"]
        if direction == "forward":
            return self.get_block(current + step_size)
        elif direction == "backward":
            return self.get_block(current - step_size)


