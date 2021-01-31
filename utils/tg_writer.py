"""Helper class to send data to telegram channel.

To use:
1. Create bot via BotFather
2. Create channel
3. Add the bot to the channel.
I've followed this explanation: https://stackoverflow.com/questions/33858927/how-to-obtain-the-chat-id-of-a-private-telegram-channel
4. Add to the root folder of this repository .ENV file, after that
add 2 environment variables to it:
    TELEGRAM_BOT_TOKEN="<replace with bot secret token>"
    TELEGRAM_CHANNEL="<replace with your channel id>"

To test, add some example.png file to the root folder and run:

    ./runner.sh python utils/tg_writer.py

The script also can send .gif or .mp4 file.
"""
import os
import telegram


class TelegramPost(object):
    def __init__(self):
        self.text = []
        self.media = None
        self.tags = []

    def add_text(self, text):
        self.text.append(text)

    def add_param(self, key, value):
        self.text.append(f"{key}: {value}")

    def add_media(self, media):
        self.media = media  # currently support only 1 photo or gif in posts

    def set_tags(self, tags):
        self.tags = tags

    def send(self, bot, channel, experiment_tags=None):
        if bot is None:
            return
        text = [str(x) for x in self.text]
        if experiment_tags is not None and isinstance(experiment_tags, str):
            text.append(experiment_tags)
        text = "\n".join(text)
        if self.media is None:
            bot.send_message(channel, text)
        else:
            name, extension = os.path.splitext(self.media)
            if extension == ".png":
                with open(self.media, 'rb') as image:
                    bot.send_photo(
                        channel, photo=image, caption=text)
            elif extension == ".gif":
                with open(self.media, 'rb') as image:
                    bot.send_animation(
                        channel, animation=image, caption=text)
            elif extension == ".mp4":
                with open(self.media, 'rb') as image:
                    bot.send_video(
                        channel, video=image, caption=text)


class PostWrapper(object):
    def __init__(self, bot, channel, tags=None):
        self.post = None
        self.bot = bot
        self.channel = channel
        self.tags = None

    def __enter__(self):
        self.post = TelegramPost()
        return self.post

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        self.post.send(self.bot, self.channel, experiment_tags=self.tags)


class TelegramWriter(object):
    def __init__(self, token=None, channel=None, tags=None):
        self.token = token
        self.channel = channel
        self.tags = tags
        if token is not None:
            self.bot = telegram.Bot(token=token)
        else:
            self.bot = None

    def post(self):
        return PostWrapper(self.bot, self.channel, self.tags)


if __name__=="__main__":
    # sample run:
    token = os.environ.get('TELEGRAM_BOT_TOKEN', None)
    channel = os.environ.get('TELEGRAM_CHANNEL', None)
    writer = TelegramWriter(token, channel)
    with writer.post() as f:
        print(f)
        f.add_text("Parameters:")
        f.add_param("smth", 0.4)
        f.add_media("protein.gif")
        f.add_text("#hashtag")