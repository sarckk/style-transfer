# Neural style transfer

Implementation of neural style transfer from [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576). Support for VGG19 only.

## Project Structure

-   `app.py` is code for a Streamlit app that provides simple UI interface for running style transfer on images of the user's choice.
-   `style_transfer.py` is the main Python code.
-   `style_transfer.ipynb` is the Jupyter Notebook version.

## API

The entrypoint for running a style transfer is the `style_transfer` function from `style_transfer.py`.

It takes the following parameters:

-   `content_im`: bytes-like object representing the content image
-   `style_im`: bytes-like object representing the style image
-   `im_size`: _tuple_ representing the dimensions of the final output image or _number_ representing the max. side length of the final output image. Content and style images get resized according to this parameter.
-   `cb`: callback function which is run every 10 iterations.<br>
    The signature of the `cb` function is as follows:<br>
    `cb(current_iteration_number, num_steps, style_loss, content_loss)`
-   `num_steps`: number of iterations algorithm takes to generate the final image.
-   `content_weight`: weight that the algorithm gives to preserving the content of the content image. **Defaults to** 10.
-   `style_weight`: weight given to preserve the style of the style image. **Defaults to** 9000.
-   `content_layers`: comma-separated list of layer names used to obtain content loss. **Defaults to** `['r4_2']`.<sup id="a1">[\*](#f1)</sup>
-   `style_layers`: comma-separated list of layer names used to obtain style loss. <br>
    **Defaults to** `['r1_1','r2_1','r3_1','r4_1','r5_1']`.<sup id="a1">[\*](#f1)</sup>

<b id="f1">\*</b> Refer to the representation of _VGG19_ layers below to understand which layers to pass to `content_layers` and `style_layers`. Simply add `r` before the layer you want to target (e.g. `r1_1` for the first layer with `64` output channels):

<pre>
1_1 1_2      2_1. 2_2.      3_1. 3_2. 3_3. 3_4.      4_1. 4_2  4_3. 4_4.      5_1. 5_2. 5_3. 5_4.  
64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'
</pre>

*P* stands for pooling layers.

## Results

| Content Image                                                                                                   | Style Image                                                                                                     |                                                                                                                                                                Result |
| --------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![image](https://user-images.githubusercontent.com/48474650/120270919-4b350f80-c2e5-11eb-81a1-2bd3b297d540.png) | ![image](https://user-images.githubusercontent.com/48474650/120271052-8e8f7e00-c2e5-11eb-8746-eb47182442b0.png) | <img width="700" alt="Screenshot 2021-06-01 at 14 28 28" src="https://user-images.githubusercontent.com/48474650/120271091-a36c1180-c2e5-11eb-9838-254d95605b85.png"> |
