import os
import shutil
import splitfolders

"""Script for moving *.wav files into parent user directory
Run after downloading and unpacking *.zip file from Google Drive
"""

def move_folders(root):

    for subdir in os.listdir(root):
        if os.path.isdir(os.path.join(root, subdir)):
            for user_subdir in os.listdir(os.path.join(root, subdir)):
                current_dir = os.path.join(root, subdir, user_subdir)
                if os.path.isdir(current_dir):
                    print(len(os.listdir()))
                    if len(os.listdir()) < 14:
                        shutil.rmtree(current_dir)
                    else:
                        for filename in os.listdir(os.path.join(root, subdir, user_subdir)):
                            shutil.move(os.path.join(root, subdir, user_subdir, filename), os.path.join(root, subdir, filename))
                else:
                    pass
        else:
            pass
       
def split_train_test(root):
    splitfolders.ratio(root, output=os.path.join('split_data'), seed=1337, ratio=(.7, .3)) 

def remove_users(root):
    """Removing users with less than 15 audio files"""
    for subdir in os.listdir(root):
        if os.path.isdir(os.path.join(root, subdir)):
            current_dir = os.path.join(root, subdir)
            if len(os.listdir(current_dir)) < 15:
                shutil.rmtree(current_dir)

if __name__ == "__main__":
    """Insert your data folder here"""
    os.makedirs('split_data', exist_ok=True) # folder to save files
    root = os.path.join('data/voxceleb')
    move_folders(root)
    #remove_users(root)
    split_train_test(root)

