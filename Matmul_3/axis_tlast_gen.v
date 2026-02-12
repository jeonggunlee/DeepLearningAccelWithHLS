module axis_tlast_gen #(
    parameter integer TDATA_W     = 32,
    parameter integer FRAME_WORDS = 512
)(
    input  wire                   aclk,
    input  wire                   aresetn,

    // S_AXIS (from DMA MM2S)
    input  wire [TDATA_W-1:0]     s_axis_tdata,
    input  wire [TDATA_W/8-1:0]   s_axis_tkeep,
    input  wire                   s_axis_tvalid,
    output wire                   s_axis_tready,
    input  wire                   s_axis_tlast,   // ignored

    // M_AXIS (to HLS s_in)
    output wire [TDATA_W-1:0]     m_axis_tdata,
    output wire [TDATA_W/8-1:0]   m_axis_tkeep,
    output wire                   m_axis_tvalid,
    input  wire                   m_axis_tready,
    output wire                   m_axis_tlast
);

    // Pass-through
    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tkeep  = s_axis_tkeep;
    assign m_axis_tvalid = s_axis_tvalid;

    // Ready propagation
    assign s_axis_tready = m_axis_tready;

    // Count only when transfer happens
    wire xfer = s_axis_tvalid && s_axis_tready;

    localparam integer CNT_W = $clog2(FRAME_WORDS);
    reg [CNT_W-1:0] beat_cnt;

    // Assert TLAST exactly at last beat of frame
    assign m_axis_tlast = xfer && (beat_cnt == FRAME_WORDS-1);

    always @(posedge aclk) begin
        if (!aresetn) begin
            beat_cnt <= {CNT_W{1'b0}};
        end else if (xfer) begin
            if (beat_cnt == FRAME_WORDS-1)
                beat_cnt <= {CNT_W{1'b0}};
            else
                beat_cnt <= beat_cnt + 1'b1;
        end
    end

endmodule
