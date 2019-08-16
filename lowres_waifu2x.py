
# MIT License
#
# Copyright (c) 2019 Geoffrey Tang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import os
import PIL
import PIL.Image
import PIL.ImageFilter
import random
import shutil
import subprocess
import sys
import threading
import time

__all__ = [
    'lowres_waifu2x',
]


def mdotmax(a, b):
    """mdotmax(a, b): max(a[..], b[..]) for matrices"""
    a_ = a[0] + a[1] + a[2]
    b_ = b[0] + b[1] + b[2]
    m = a_ >= b_
    n = a_ < b_
    c = list((m * a[i] + n * b[i]) for i in range(0, 3))
    return c


def mdotmin(a, b):
    """mdotmin(a, b): min(a[..], b[..]) for matrices"""
    a_ = a[0] + a[1] + a[2]
    b_ = b[0] + b[1] + b[2]
    m = a_ <= b_
    n = a_ > b_
    c = list((m * a[i] + n * b[i]) for i in range(0, 3))
    return c


def mdotadd(a, b):
    """mdotadd(a, b): a[..] + b[..] for matrices"""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def mdotmul(a, b):
    """mdotmul(a, b): a[..] .* b[..] for matrices"""
    return [a[0] * b, a[1] * b, a[2] * b]


def msigmoid(x):
    """msigmoid(x): modified sigmoid function"""
    return (1 / (1 + np.e ** (- 14.4 * (x - 0.35))) - 0.5) * 0.99 + 0.5


def disp_progress(info, done=None, sigma=None, begin=None):
    """disp_progress(info, *done, *sigma, *begin): display progress info
    ...(info): display info only,
    ...(info, True): display info and mark as done
    ...(info, done, sigma, begin): done / sigma jobs completed, all jobs
                                   began at time 'begin' """
    if done is None:
        print('%s... %s\r' % (info, ' ' * 20), end='')
        sys.stdout.flush()
        return
    elif done is True:
        print('%s... done%s' % (info, ' ' * 20))
        sys.stdout.flush()
        return
    if done == 0:
        tm = '---'
    else:
        tm = '%.2fs' % ((time.time() - begin) * (sigma - done) / done)
    prc = done / sigma * 100.0
    print('%s... %.2f%% (eta %s)%s\r' % (info, prc, tm, ' ' * 20), end='')
    sys.stdout.flush()
    return


def lowres_waifu2x(waifu2x_path):
    """lowres_waifu2x(waifu2x_path): execute enhanced waifu2x"""
    do_presave = True
    do_supersample = True
    do_filter = True
    ###########################################################################
    # Undersample images to a certain range
    ###########################################################################
    if do_presave:
        rng = (0.45, 0.95)
        img = PIL.Image.open('./source.png', mode='r')
        area = img.width * img.height
        if area <= 262144:
            cnt = 48
        elif area <= 786432:
            cnt = 32
        elif area <= 2097152:
            cnt = 16
        else:
            cnt = 8
        print('using a kernel size of %d.' % cnt)
        t1 = time.time()
        disp_progress('undersampling images', 0, 2 * cnt + 2, t1)
        img.save('./ssample-src-dn-0.png')
        disp_progress('undersampling images', 1, 2 * cnt + 2, t1)
        img.save('./ssample-src-or-0.png')
        for i in range(1, cnt + 1):
            disp_progress('undersampling images', i * 2, 2 * cnt + 2, t1)
            ratio = random.random() * (rng[1] - rng[0]) + rng[0]
            size = (img.width * ratio, img.height * ratio)
            img.resize(map(int, size)).save('./ssample-src-dn-%d.png' % i)
            disp_progress('undersampling images', i * 2 + 1, 2 * cnt + 2, t1)
            ratio = random.random() * (rng[1] - rng[0]) + rng[0]
            size = (img.width * ratio, img.height * ratio)
            img.resize(map(int, size)).save('./ssample-src-or-%d.png' % i)
        disp_progress('undersampling images', True)
    ###########################################################################
    # Supersample undersampled images to target size
    ###########################################################################
    if do_supersample:
        t1 = time.time()
        # Create directories for buffer
        shutil.rmtree('./waifu2x-input-dn', ignore_errors=True)
        shutil.rmtree('./waifu2x-input-or', ignore_errors=True)
        shutil.rmtree('./waifu2x-output', ignore_errors=True)
        os.mkdir('./waifu2x-input-dn')
        os.mkdir('./waifu2x-input-or')
        os.mkdir('./waifu2x-output')
        # Copy images output to buffer
        for i in range(0, cnt + 1):
            disp_progress('buffering images', i * 2, 2 * cnt + 2, t1)
            shutil.copy('./ssample-src-dn-%d.png' % i, './waifu2x-input-dn/')
            disp_progress('buffering images', i * 2 + 1, 2 * cnt + 2, t1)
            shutil.copy('./ssample-src-or-%d.png' % i, './waifu2x-input-or/')
        disp_progress('buffering images', True)

        def worker():
            proc = subprocess.Popen(
                args=[waifu2x_path,
                      '--scale_ratio', '4',
                      '--noise_level', '2',
                      '--input_path', './waifu2x-input-dn/',
                      '--output_path', './waifu2x-output/'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.wait()
            proc = subprocess.Popen(
                args=[waifu2x_path,
                      '--scale_ratio', '4',
                      '--noise_level', '0',
                      '--input_path', './waifu2x-input-or/',
                      '--output_path', './waifu2x-output/'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.wait()
            return
        # Monitor output directory
        workerd = threading.Thread(target=worker, args=[])
        workerd.start()
        t1 = time.time()
        wo_cnt = 0
        tot = 2 * cnt + 2
        while wo_cnt < tot:
            time.sleep(0.1)
            wo_cnt = len(os.listdir('./waifu2x-output/'))
            disp_progress('supersampling images', wo_cnt, tot, t1)
        workerd.join()
        disp_progress('supersampling images', True)
        # Retrieve buffer back to current directory
        for i in range(0, cnt + 1):
            disp_progress('retrieving output', i * 2, 2 * cnt + 2, t1)
            shutil.copy('./waifu2x-output/ssample-src-dn-%d.png' % i,
                        './ssample-out-dn-%d.png' % i)
            disp_progress('retrieving output', i * 2 + 1, 2 * cnt + 2, t1)
            shutil.copy('./waifu2x-output/ssample-src-or-%d.png' % i,
                        './ssample-out-or-%d.png' % i)
        # Remove buffer
        shutil.rmtree('./waifu2x-input-dn', ignore_errors=True)
        shutil.rmtree('./waifu2x-input-or', ignore_errors=True)
        shutil.rmtree('./waifu2x-output', ignore_errors=True)
        disp_progress('retrieving output', True)
    ###########################################################################
    # Load original image
    ###########################################################################
    orig_img = PIL.Image.open('./source.png', mode='r')
    size = (orig_img.width * 4, orig_img.height * 4)
    orig_img = orig_img.resize(size)

    def load_image(fn, size):
        img = PIL.Image.open(fn, mode='r')
        img = img.resize(size).convert('RGB')
        mat = np.array(img)
        mat = mat.transpose((2, 0, 1)) * np.ones((3, size[1], size[0]))
        mat = [mat[0], mat[1], mat[2]]
        return mat
    orig_mat = load_image('./source.png', size)
    ###########################################################################
    # Image filtering
    ###########################################################################
    if do_filter:
        t1 = time.time()
        rsize = (size[1], size[0])
        dnm1 = orig_mat
        dnm2 = orig_mat
        dnm3 = [np.zeros(rsize), np.zeros(rsize), np.zeros(rsize)]
        orm1 = orig_mat
        orm2 = orig_mat
        orm3 = [np.zeros(rsize), np.zeros(rsize), np.zeros(rsize)]
        for i in range(0, cnt + 1):
            disp_progress('filtering matrices', i * 2, 2 * cnt + 2, t1)
            mat = load_image('./ssample-out-dn-%d.png' % i, size)
            dnm1 = mdotmax(dnm1, mat)
            dnm2 = mdotmin(dnm2, mat)
            dnm3 = mdotadd(dnm3, mat)
            disp_progress('filtering matrices', i * 2 + 1, 2 * cnt + 2, t1)
            mat = load_image('./ssample-out-or-%d.png' % i, size)
            orm1 = mdotmax(orm1, mat)
            orm2 = mdotmin(orm2, mat)
            orm3 = mdotadd(orm3, mat)
        disp_progress('filtering matrices', True)
        # Combine results
        disp_progress('combining results')
        dnm3 = mdotmul(dnm3, 1 / (cnt + 1))
        dnm = mdotadd(mdotmul(dnm1, 0.8), mdotmul(dnm2, 0.2))
        dnm = mdotadd(mdotmul(dnm, 0.6), mdotmul(dnm3, 0.4))
        orm3 = mdotmul(orm3, 1 / (cnt + 1))
        orm = mdotadd(mdotmul(orm1, 0.25), mdotmul(orm2, 0.75))
        orm = mdotadd(mdotmul(orm, 0.75), mdotmul(orm3, 0.25))
        ratio = msigmoid((orm[0] + orm[1] + orm[2]) / (255 * 3))
        d = mdotadd(mdotmul(dnm, ratio), mdotmul(orm, 1 - ratio))
        disp_progress('combining results', True)
        # Transpose and export
        disp_progress('exporting matrix')
        d = np.array(d)
        d = d.transpose((1, 2, 0))
        orig_img = PIL.Image.fromarray(np.uint8(d))
        disp_progress('exporting matrix', True)
    # Kernel filter
    disp_progress('saving output')
    nsize = (size[0] // 2, size[1] // 2)
    orig_img = orig_img.resize(nsize)
    kernel_module = (
        -1, -7,  -9, -7, -1,
        -7, -2,  -4, -2, -7,
        -9, -4, 384, -4, -9,
        -7, -2,  -4, -2, -7,
        -1, -7,  -9, -7, -1,
    )  # Sharpen filter
    kernel_filter = PIL.ImageFilter.Kernel(
        size=(5, 5),
        kernel=kernel_module,
        scale=sum(kernel_module),
        offset=0,
    )
    orig_img = (orig_img.filter(kernel_filter)
                .filter(PIL.ImageFilter.SHARPEN))
    # Save file
    orig_img.save('./output.png')
    # Cleanup
    for i in range(0, cnt + 1):
        shutil.rmtree('./ssample-out-dn-%d.png' % i, ignore_errors=False)
        shutil.rmtree('./ssample-out-or-%d.png' % i, ignore_errors=False)
    disp_progress('saving output', True)
    return


if __name__ == '__main__':
    waifu2x_path = './waifu2x/waifu2x-caffe-cui.exe'
    if len(sys.argv) >= 2:
        waifu2x_path = sys.argv[1]
    if not os.path.exists(waifu2x_path):
        print('lowres_waifu2x: waifu2x binary does not exist')
        print('arguments: lowres_waifu2x path_to_waifu2x-caffe-cui')
        print('transformation terminated.')
        exit(1)
    lowres_waifu2x(waifu2x_path)
