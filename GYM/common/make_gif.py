import matplotlib.pyplot as plt
import matplotlib.animation as anim
from typing import Optional

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames: list, save_path: str, title: Optional[str] = None, repeat=False, interval=500):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    if title is None:
        title = save_path
    plt.title(title, fontsize=16)
    animation = anim.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    animation.save(save_path, writer='pillow', fps=20)
    return animation