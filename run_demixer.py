"""
Interactive masknmf demixing on the GPU: superpixel initialization, demixing passes viewed while they run, and
initialization again on the residual of a pass, its signals appended to those of the pass. initialize signals and demix
signals use masknmf's defaults.

On top: the frames of masknmf's PMD array, the local correlation image of the last initialization with its superpixels
(red) and pure superpixels (green), and the frames of masknmf's AC and residual arrays, all without rescaling. Double
click a signal in the PMD, AC or residual image to show its trace below, outside the signals to clear it.
"""

import numpy as np
import torch
import fastplotlib as fpl
from fastplotlib.ui import ImguiColorbar
from imgui_bundle import imgui
from masknmf.demixing import NoSignalsDetectedError
from masknmf.demixing.demixing_arrays.ac_array import ACArray
from masknmf.visualization.imgui import component_at_pixel

from project import select_adapter, load_compression, CompressionBuffers
from project._demixer import Demixer
from project._viewer import DemixingFrames

# if WGPU can't find an adapter it will raise, or it will give you a LLVM CPU adapter which you don't want
# if you have multiple GPUs you can select it using the int index, here the GPU at index 0 is selected.
adapter = fpl.enumerate_adapters()[0]
print(adapter.info)
select_adapter(adapter)

# path to the demixing results file, only the compression results are used
dmr_path = "./demix_new.hdf5"
# frames per second of the movie, the traces are in seconds
frame_rate = 5.076

compression = load_compression(dmr_path)
n_frames, height, width = compression["shape"]
gpu_compression = CompressionBuffers(dmr_path)
demixer = Demixer(gpu_compression, compression)
frames = DemixingFrames(gpu_compression)

image_names = ["pmd", "initialized", "ac", "residual"]
figure = fpl.Figure(
    extents={
        **{name: (i / 4, (i + 1) / 4, 0, 0.55) for i, name in enumerate(image_names)},
        "traces": (0, 1, 0.55, 1),
    },
    controller_ids=[tuple(image_names)],
    size=(1800, 1000),
)

# the PMD, AC and residual frames, each with its colorbar
vmin, vmax = frames.image_graphics[0].vmin, frames.image_graphics[0].vmax
for name, image in zip(["pmd", "ac", "residual"], frames.image_graphics):
    figure[name].add_graphic(image)
    figure[name].add_imgui_window(ImguiColorbar(image, data_range=(vmin, vmax)), location="right", size=80)

# the contours of the signals, the selected one highlighted
highlight = fpl.ImageHighlightSelector()
for image in frames.image_graphics:
    highlight.add_graphic(image)

# the trace of the selected signal, in the color of its contour. The line is created at the first selection: pygfx does
# not change the revision of a buffer that is already marked for upload, so a line updated before it was ever drawn
# keeps the bounding box of its first data, which auto_scale would fit
x = (np.arange(n_frames) / frame_rate).astype(np.float32)
trace = None
figure["traces"].camera.maintain_aspect = False

# the local correlation image of the last initialization and the markers of its superpixels, from the first one
correlation_image = None
superpixel_markers = None
pure_markers = None

# the signals shown: of initialize_signals before a pass, of the demixer during and after it
signals = None
# masknmf's ACArray of the signals for the contours and the centers
rois = None
selected = None
# the demix pass while it runs, and the action requested from the controls
iterations = None
request = None
status = "initialize signals to start"
t = 0


def select(signal: int | None):
    """highlight the contour of a signal and show its trace, None for no signal"""
    global selected, trace
    selected = signal
    highlight.selection = [] if signal is None else [signal]
    if signal is None:
        if trace is not None:
            trace.visible = False
    elif trace is None:
        trace = figure["traces"].add_line(np.column_stack([x, signals.get_trace(signal)]), colors="red")
    else:
        trace.visible = True
        trace.data[:, 1] = signals.get_trace(signal)


def update_rois():
    """masknmf's contours and centers of the signals, which clears the selection of the contours"""
    global rois
    rois = ACArray.from_tensors((height, width), signals.get_a(), torch.from_numpy(signals.get_c()))
    highlight.selection_options = {"pixels": rois.contours}


def show_signals(new_signals):
    """new signals in the frames and the contours, the selection cleared"""
    global signals
    signals = new_signals
    frames.set_signals(signals)
    update_rois()
    select(None)


def show_superpixels():
    """the local correlation image of the last initialization, its superpixels and pure superpixels"""
    global correlation_image, superpixel_markers, pure_markers
    image, superpixels, pure_superpixels = demixer.get_superpixels()
    # (x, y) = (column, row)
    positions = [np.column_stack([p % width, p // width]).astype(np.float32) for p in (superpixels, pure_superpixels)]
    subplot = figure["initialized"]
    if correlation_image is None:
        correlation_image = subplot.add_image(image, cmap="gray")
        subplot.add_imgui_window(ImguiColorbar(correlation_image), location="right", size=80)
        superpixel_markers = subplot.add_scatter(positions[0], colors="red", sizes=4)
        pure_markers = subplot.add_scatter(positions[1], colors="lime", sizes=4)
        # the subplot was empty when the figure was shown, its camera starts at the view of the others
        subplot.camera.set_state(figure["pmd"].camera.get_state())
    else:
        correlation_image.data = image
        correlation_image.reset_vmin_vmax()
        superpixel_markers.data = positions[0]
        pure_markers.data = positions[1]


def step():
    """one iteration of the pass, new signals shown when they changed, the contours again after the last iteration"""
    global iterations, status
    try:
        iteration = next(iterations)
    except StopIteration:
        iterations = None
        update_rois()
        select(selected)
        status = f"pass done, {signals.n_signals} signals"
        return
    except ValueError as e:
        iterations = None
        status = f"demix stopped: {e}"
        return

    if demixer.signals is signals:
        frames.set_signals(signals)
        if selected is not None:
            trace.data[:, 1] = signals.get_trace(selected)
    else:
        show_signals(demixer.signals)
    status = f"iteration {iteration}, {signals.n_signals} signals, background rank {demixer.background_rank}"


def pick_signal(ev):
    """masknmf's picking: the signal whose footprint contains the pixel, of those the one with the nearest center"""
    if rois is None:
        return
    select(component_at_pixel(rois.a, rois.centers, (height, width), ev.pick_info["index"]))
    if selected is not None:
        figure["traces"].auto_scale(maintain_aspect=False)


for image in frames.image_graphics:
    image.add_event_handler(pick_signal, "double_click")


@figure.add_imgui_window(location="bottom", size=70)
def controls(figure):
    global request, t
    imgui.begin_disabled(iterations is not None)
    if imgui.button("initialize signals"):
        request = "initialize"
    imgui.same_line()
    imgui.begin_disabled(signals is None)
    if imgui.button("demix signals"):
        request = "demix"
    imgui.end_disabled()
    imgui.end_disabled()
    imgui.same_line()
    imgui.text(status)
    imgui.set_next_item_width(-1)
    _, t = imgui.slider_int("##frame", t, 0, n_frames - 1, "frame %d")


def update():
    """the requested action, one iteration of a running pass, frame t"""
    global request, iterations, status
    if request == "initialize":
        try:
            show_signals(demixer.initialize_signals())
            show_superpixels()
            status = f"initialized, {signals.n_signals} signals"
        except NoSignalsDetectedError as e:
            status = f"no new signals: {e}"
    elif request == "demix":
        iterations = demixer.demix(signals)
        status = "demixing"
    request = None

    if iterations is not None:
        step()
    if frames.t != t:
        frames.t = t


figure.add_animations(update)
figure.show()

if __name__ == "__main__":
    fpl.loop.run()
