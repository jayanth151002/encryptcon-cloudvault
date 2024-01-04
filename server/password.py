import hashlib
import os
import random
import string


class PasswordManager:
    def __init__(self):
        pass

    def generate_password(self, length=None):
        if length is None:
            length = random.randint(8, 15)

        characters = string.ascii_letters + string.digits + string.punctuation
        while True:
            password = "".join(random.choice(characters) for _ in range(length))
            if (
                any(c.islower() for c in password)
                and any(c.isupper() for c in password)
                and any(c.isdigit() for c in password)
                and any(c in string.punctuation for c in password)
            ):
                return password

    def generate_salt(self):
        return os.urandom(16)

    def hash_password(self, password, salt):
        hashed_password = hashlib.sha256(password.encode("utf-8") + salt).hexdigest()
        return hashed_password

    def encode_password(self, password):
        salt = self.generate_salt()
        hashed_password = self.hash_password(password, salt)
        return hashed_password, salt

    def verify_password(self, entered_password, stored_password, salt):
        entered_password_hash = self.hash_password(entered_password, salt)
        return entered_password_hash == stored_password
