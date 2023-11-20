import os
import jax
jax.devices()
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess

import jax
import jax.numpy as jnp
import ml_collections

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,9"


# Pick your hero: (WHEN CHANGING THIS, RERUN IMAGE/TEXT EMBEDDING CELLS)
# Give this cell 1-3mins.

# VARIANT, RES = 'B/16', 224
VARIANT, RES = 'B/16', 256
# VARIANT, RES = 'B/16', 384
# VARIANT, RES = 'B/16', 512
# VARIANT, RES = 'L/16', 256
# VARIANT, RES = 'L/16', 384
# VARIANT, RES = 'So400m/14', 224
# VARIANT, RES = 'So400m/14', 384
# VARIANT, RES = 'B/16-i18n', 256

CKPT, TXTVARIANT, EMBDIM, SEQLEN, VOCAB = {
    ('B/16', 224): ('webli_en_b16_224_63724782.npz', 'B', 768, 64, 32_000),
    ('B/16', 256): ('webli_en_b16_256_60500360.npz', 'B', 768, 64, 32_000),
    ('B/16', 384): ('webli_en_b16_384_68578854.npz', 'B', 768, 64, 32_000),
    ('B/16', 512): ('webli_en_b16_512_68580893.npz', 'B', 768, 64, 32_000),
    ('L/16', 256): ('webli_en_l16_256_60552751.npz', 'L', 1024, 64, 32_000),
    ('L/16', 384): ('webli_en_l16_384_63634585.npz', 'L', 1024, 64, 32_000),
    ('So400m/14', 224): ('webli_en_so400m_224_57633886.npz', 'So400m', 1152, 16, 32_000),
    ('So400m/14', 384): ('webli_en_so400m_384_58765454.npz', 'So400m', 1152, 64, 32_000),
    ('B/16-i18n', 256): ('webli_i18n_b16_256_66117334.npz', 'B', 768, 64, 250_000),
}[VARIANT, RES]

file_path = f"./models/siglip/{CKPT}"
# Check if the file exists
if not os.path.isfile(file_path):
    print("downloading file")
    # If the file doesn't exist, copy it from Google Cloud Storage
    subprocess.run(['gsutil', 'cp', f'gs://big_vision/siglip/{CKPT}', file_path], check=True)
else:
    print("file exists")

sys.path.append('/home/shang/vidVQA/models/big_vision/')
import big_vision.models.proj.image_text.two_towers as model_mod

model_cfg = ml_collections.ConfigDict()
model_cfg.image_model = 'vit'  # TODO(lbeyer): remove later, default
model_cfg.text_model = 'proj.image_text.text_transformer'  # TODO(lbeyer): remove later, default
model_cfg.image = dict(variant=VARIANT, pool_type='map')
model_cfg.text = dict(variant=TXTVARIANT, vocab_size=VOCAB)
model_cfg.out_dim = (None, EMBDIM)  # (image_out_dim, text_out_dim)
model_cfg.bias_init = -10.0
model_cfg.temperature_init = 10.0

model = model_mod.Model(**model_cfg)

# Using `init_params` is slower but will lead to `load` below performing sanity-checks.
# init_params = jax.jit(model.init, backend="cpu")(jax.random.PRNGKey(42), jnp.zeros([1, RES, RES, 3], jnp.float32), jnp.zeros([1, SEQLEN], jnp.int32))['params']
init_params = None  # Faster but bypasses loading sanity-checks.

params = model_mod.load(init_params, f'./models/siglip/{CKPT}', model_cfg)

import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_general
import big_vision.pp.ops_image
import big_vision.pp.ops_text
import PIL

#@title Load and embed images
images = [PIL.Image.open("./data/" + fname) for fname in (
    'apple-ipod.jpg',
    'apple-blank.jpg',
    'cold_drink.jpg',
    'hot_drink.jpg',
    'caffeine.jpg',
    'siglip.jpg',
    'authors.jpg',
    'robosign.jpg',
    'cow_beach.jpg',
    'cow_beach2.jpg',
    'mountain_view.jpg',
)]

pp_img = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(-1, 1)')
imgs = np.array([pp_img({'image': np.array(image)})['image'] for image in images])
zimg, _, out = model.apply({'params': params}, imgs, None)

print(imgs.shape, zimg.shape)

#@title Tokenize and embed texts

texts = [
    'an apple',
    'a picture of an apple',
    'an ipod',
    'granny smith',
    'an apple with a note saying "ipod"',
    'a cold drink on a hot day',
    'a hot drink on a cold day',
    'a photo of a cold drink on a hot day',
    'a photo of a hot drink on a cold day',
    #
    'a photo of two guys in need of caffeine',
    'a photo of two guys in need of water',
    'a photo of the SigLIP authors',
    'a photo of a rock band',
    'a photo of researchers at Google Brain',
    'a photo of researchers at OpenAI',
    #
    'a robot on a sign',
    'a photo of a robot on a sign',
    'an empty street',
    'autumn in Toronto',
    'a photo of autumn in Toronto',
    'a photo of Toronto in autumn',
    'a photo of Toronto in summer',
    'autumn in Singapore',
    #
    'cow',
    'a cow in a tuxedo',
    'a cow on the beach',
    'a cow in the prairie',
    #
    'the real mountain view',
    'Zürich',
    'San Francisco',
    'a picture of a laptop with the lockscreen on, a cup of cappucino, salt and pepper grinders. The view through the window reveals lake Zürich and the Alps in the background of the city.',
]

TOKENIZERS = {
    32_000: 'c4_en',
    250_000: 'mc4',
}
pp_txt = pp_builder.get_preprocess_fn(f'tokenize(max_len={SEQLEN}, model="{TOKENIZERS[VOCAB]}", eos="sticky", pad_value=1, inkey="text")')
txts = np.array([pp_txt({'text': text})['labels'] for text in texts])
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
_, ztxt, out = model.apply({'params': params}, None, txts)

print(txts.shape, ztxt.shape)