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

The script also can send .gif files.
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

    def send(self, bot, channel):
        text = "\n".join([str(x) for x in self.text])
        
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


class PostWrapper(object):
    def __init__(self, bot, channel):
        self.post = None
        self.bot = bot
        self.channel = channel

    def __enter__(self):
        self.post = TelegramPost()
        return self.post

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        self.post.send(self.bot, self.channel)


class TelegramWriter(object):
    def __init__(self, token, channel):
        self.token = token
        self.channel = channel
        self.bot = telegram.Bot(token=token)

    def post(self):
        return PostWrapper(self.bot, self.channel)


if __name__=="__main__":
    # sample run:
    token = os.environ.get('TELEGRAM_BOT_TOKEN', None)
    channel = os.environ.get('TELEGRAM_CHANNEL', None)
    writer = TelegramWriter(token, channel)
    with writer.post() as f:
        print(f)
        f.add_text("Parameters:")
        f.add_param("smth", 0.4)
        f.add_media("example.png")
        f.add_text("#hashtag")