import argparse
import cv2
import matplotlib
import numpy as np
import os
from utils.tg_writer import TelegramWriter


def save_frames_as_gif(image_paths, path='./', name='animation'):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    frames = []
    for image_file in image_paths:
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        img = img[50+28:h-40, 200:w-200]
        h, w, c = img.shape
        h = h//2
        w = w//2
        img = cv2.resize(img, (w, h))
        frames.append(img)

    #Mess with this to change frame size
    w, h = (frames[0].shape[1] / 100., frames[0].shape[0] / 100.)
    # h=3
    # w=4
    print(h, w)
    fig = plt.figure(figsize=(w, h))

    # set figure background opacity (alpha) to 0
    #fig.patch.set_alpha(0.)
    print(img.shape)
    ax = fig.gca()

    patch = ax.imshow(frames[0], aspect='auto')
    ax.margins(0, 0)
    #ax.set_xlim(0., 600)
    #ax.set_ylim(0., 600)
    # turn off axis spines
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.set_size_inches(w, h, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    #plt.axis('off')
    #plt.savefig("test.png", pad_inches=0, bbox_inches = 'tight')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=10)
    gif_path = os.path.join(path, name + ".gif")
    # anim.save(gif_path, writer='imagemagick', fps=10, dpi=150)
    #, pad_inches=0, bbox_inches = 'tight')
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='lacemaker'), bitrate=1800)
    video_path = os.path.join(path, name + ".mp4")
    anim.save(video_path, writer=writer)

    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="saves/0000000002")
    parser.add_argument("--tg", action="store_true", default=True)
    parser.add_argument("--tag", action="append", nargs="+", default=[], type=str)
    parser.add_argument("--episode", default=0, type=int)
    parser.add_argument("--score", default=[], action="append", nargs="+", type=float)
    parser.add_argument("--size", default=0, type=int)

    args = parser.parse_args()
    tagline = " ".join(["#"+tag for tag in np.asarray(args.tag).flatten()])
    datadir = args.datadir
    if not os.path.exists(datadir):
        print(f"{datadir} was not found, exiting")
        exit(1)
    image_files = [
        (int(file.split("_")[-1].split(".")[0]), os.path.join(datadir, file))
        for file in os.listdir(datadir)
        if os.path.splitext(file)[-1] == ".png"
    ]
    image_files = sorted(image_files, key=lambda x: x[0])
    image_files = [image for k, image in image_files]
    if len(image_files) < 5:
        print("not enough image files were found, exiting")
        exit(1)
    video_path = save_frames_as_gif(image_files, path=datadir)
    score = list(np.asarray(args.score).flatten())
    if args.tg:
        # send to telegram
        token = os.environ.get('TELEGRAM_BOT_TOKEN', None)
        channel = os.environ.get('TELEGRAM_CHANNEL', None)
        writer = TelegramWriter(token, channel)

        with writer.post() as f:
            f.add_param("Episode", args.episode)
            f.add_param("Score", score)
            f.add_param("Memory used", args.size)
            f.add_text(tagline)
            f.add_media(video_path)
