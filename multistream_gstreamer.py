import sys
from turtle import width
from typing import List, Tuple
import math

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

import sys

sys.path.append('/opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/apps/')
sys.path.append('/opt/nvidia/deepstream/deepstream-6.0/lib/')
sys.path.append("/opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/bindings/build")
sys.path.append("/opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/3rdparty/gst-python")

import numpy as np
import cv2
import pyds

from utils import decodebin_child_added, cb_newpad


class MultiStreamGReader:
    def __init__(self, sources: List[str]) -> None:
        self.sources = sources
        self.frame = None
        GObject.threads_init()
        Gst.init(None)

    def start(self, size, tiled_output_size):
        pipeline, loop = self.build_pipeline(self.sources, size, tiled_output_size)
        self.run_pipeline(pipeline, loop)
        
    def read(self):
        return self.frame

    def build_pipeline(self, sources, size: Tuple[int, int], tiled_output_size):
        """
        Build gstreamer pipeline
        
        Args:
            sources: List of sources (links or files)
            size: (height, width) of streammux
            tiled_output_size: (height, width) of tiler
        
        Return:
            Builded and fully linked pipeline
        """
        height, width = size
        pipeline = Gst.Pipeline()
        # self.mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        self.mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux = self.build_streammux(
            pipeline=pipeline, sources=sources, width=width, height=height
        )
        queue1 = self.build_queue(pipeline, queue_name="queue1")
        queue2 = self.build_queue(pipeline, queue_name="queue2")
        queue3 = self.build_queue(pipeline, queue_name="queue3")
        queue4 = self.build_queue(pipeline, queue_name="queue4")

        tiled_output_height, tiled_output_width = tiled_output_size
        
        tiler = self.build_tiler(
            pipeline, 
            number_sources=len(sources),
            tiled_output_width=tiled_output_width,
            tiled_output_height=tiled_output_height
        )
        
        nvvidconv = self.build_nvvidconv(pipeline)
        filter1 = self.build_filter(pipeline, use_gpu=True, filter_name="filter1")
        filter2 = self.build_filter(pipeline, use_gpu=True, filter_name="filter2")
        # filter2 = self.build_filter(pipeline, use_gpu=False)
        sink = self.build_sink(pipeline)

        # streammux.link(queue1)        
        # queue1.link(tiler)
        # tiler.link(queue2)
        # queue2.link(nvvidconv)
        # nvvidconv.link(queue3)
        # queue3.link(filter)
        # filter.link(queue4)
        # queue4.link(sink)
        
        streammux.link(queue1)
        queue1.link(nvvidconv)
        nvvidconv.link(queue2)
        queue2.link(filter1)
        filter1.link(queue3)
        queue3.link(tiler)
        tiler.link(queue4)
        queue4.link(filter2)
        filter2.link(sink)
        
            # create an event loop and feed gstreamer bus mesages to it
        loop = GObject.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        tiler_sink_pad = queue4.get_static_pad("src")
        if not tiler_sink_pad:
            print("ASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDFASDF")
            sys.stderr.write(" Unable to get src pad \n")
        else:
            tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)
            
        return pipeline, loop
        
    def run_pipeline(self, pipeline, loop):
        print("Starting pipeline \n")
        # start play back and listed to events		
        pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass
        # cleanup
        print("Exiting app\n")
        pipeline.set_state(Gst.State.NULL)
        
    def build_filter(self, pipeline, use_gpu: bool, filter_name):
        """isn't used but could be helpful in the future"""
        use_gpu_memory = "(memory:NVMM)"
        caps_string = f"video/x-raw{use_gpu_memory * use_gpu},format=RGBA"
        caps = Gst.Caps.from_string(caps_string)
        filter = Gst.ElementFactory.make("capsfilter", filter_name)
        if not filter:
            sys.stderr.write(" Unable to get the caps filter \n")
        filter.set_property("caps", caps)
        pipeline.add(filter)
        return filter
        
    def build_nvosd(self, pipeline, osd_process_mode, osd_display_text):
        """Using to draw rectangles and text. NOTE: Useless for our pipeline"""
        print("Creating nvosd \n ")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")
        nvosd.set_property('process-mode', osd_process_mode)
        nvosd.set_property('display-text', osd_display_text)
        pipeline.add(nvosd)
        return nvosd
        
    def build_sink(self, pipeline):
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")
        pipeline.add(sink)
        return sink
        
    def build_nvvidconv(self, pipeline):
        print("Creating nvvidconv \n ")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")
        nvvidconv.set_property("nvbuf-memory-type", self.mem_type)
        pipeline.add(nvvidconv)
        return nvvidconv
        
    def build_tiler(self, pipeline, number_sources, tiled_output_width, tiled_output_height):
        print("Creating tiler \n ")
        tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            sys.stderr.write(" Unable to create tiler \n")
        tiler_rows=int(math.sqrt(number_sources))
        tiler_columns=int(math.ceil((1.0 * number_sources) / tiler_rows))
        tiler.set_property("nvbuf-memory-type", self.mem_type)
        tiler.set_property("rows",tiler_rows)
        tiler.set_property("columns",tiler_columns)
        tiler.set_property("width", tiled_output_width)
        tiler.set_property("height", tiled_output_height)
        pipeline.add(tiler)
        return tiler

    def build_queue(self, pipeline, queue_name):
        queue = Gst.ElementFactory.make("queue", queue_name)
        pipeline.add(queue)
        return queue

    def build_streammux(self, pipeline, sources: List, width: int, height: int):
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        is_live = False
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")
        pipeline.add(streammux)

        for num, source in enumerate(sources):
            print("Creating source_bin ", num," \n ") 
            if source.find("rtsp://") == 0 :
                is_live = True
            source_bin = self._create_source_bin(num, source)
            if not source_bin:
                sys.stderr.write("Unable to create source bin \n")
            pipeline.add(source_bin)
            padname = f"sink_{num}"
            sinkpad = streammux.get_request_pad(padname) 
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)

        if is_live:
            print("Atleast one of the sources is live")
            streammux.set_property('live-source', 1)

        streammux.set_property("nvbuf-memory-type", self.mem_type)
        streammux.set_property('width', width)
        streammux.set_property('height', height)
        streammux.set_property('batch-size', len(sources))
        streammux.set_property('batched-push-timeout', 4000000)
        return streammux

    @staticmethod
    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
        return True
    
    def _create_source_bin(self, index, uri):
        print("Creating source bin")
        bin_name="source-bin-%02d" %index
        print(bin_name)
        nbin=Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
        Gst.Bin.add(nbin,uri_decode_bin)
        bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin
    
    def tiler_sink_pad_buffer_probe(self, pad, info, u_data):
        print("uioyuioyuoiyuioyuioyuioyuioyuioyuioyuioyuioyuioyuioyuioyuio")
        frame_number = 0
        gst_buffer = info.get_buffer()
        # print(1)
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return
        # print(2)
        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        # print(3)
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                # print(4)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            # num_rects = frame_meta.num_obj_meta
            # is_first_obj = True
            save_image = True

            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            # print(5, frame_meta.batch_id)
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            # print(6)
            # convert python array into numpy array format in the copy mode.
            frame_copy = np.array(n_frame, copy=True, order='C')
            # print(7, frame_copy.shape)

            # convert the array into cv2 default color format
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
            # print(8)

            # Get frame rate through this probe
            # fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
            if save_image:
                # print(9)

                img_path = "{}/stream_{}/frame_{}.jpg".format("tmp", frame_meta.pad_index, frame_number)
                # print(img_path)
                cv2.imwrite(img_path, frame_copy)
            try:
                # print(10)
                l_frame = l_frame.next
                # print(11)
            except StopIteration:
                break
        # print(12)
        return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    import sys
    sources = sys.argv[1:]
    streamer = MultiStreamGReader(sources)
    streamer.start((720, 1080), (720, 1080))


