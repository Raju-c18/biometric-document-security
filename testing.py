from rubikencryptor.rubikencryptor import RubikCubeCrypto
from PIL import Image

key = 'key.txt'

fileName = 'original.png'
input_image = "static/original/"+fileName
output_image = "static/encrypted/"+fileName

input_image = Image.open(input_image)
rubixCrypto = RubikCubeCrypto(input_image)

encrypted_image = rubixCrypto.encrypt(alpha=8, iter_max=10, key_filename=key)
encrypted_image.save(output_image)

input_image = "static/encrypted/"+fileName
output_image = "static/decrypted/"+fileName

input_image = Image.open(input_image)
rubixCrypto = RubikCubeCrypto(input_image)

decrypted_image = rubixCrypto.decrypt(key_filename=key)
decrypted_image.save(output_image)
