def process_sam2(sam2, hwc):
    hwc = hwc.to(sam2.device)
    img = hwc.permute(2, 0, 1).unsqueeze(0)  # bchw

    # initialize
    if not sam2.is_initialized:
        print("initializing segmentor...")

        sam2.init_state(img, channel_last=False)

        sam2.annotate(flip_y=True)
        sam2.capture_cuda_graph(img)
        print("initializing segmentor done")
    else:
        sam2.add_image(img)
        sam2.propagate()

    object_masks = sam2.get_combined_object_masks()

    img = object_masks.unsqueeze(0)  # b1hw

    return img