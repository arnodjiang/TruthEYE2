def extract_images_from_messages(messages):
    files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.image and item.image not in files:
                    files.append(item.image)
    return files