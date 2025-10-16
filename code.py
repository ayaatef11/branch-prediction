
import tkinter as tk
from tkinter import ttk
import random
#factory pattern
class Instruction:
    def __init__(self, name, reads=None, writes=None, is_branch=False):
        self.name = name
        self.reads = reads or []    
        self.writes = writes or [] 
        self.is_branch = is_branch
        self.predicted_taken = None
        self.actual_taken = None

class InstructionFactory:
    @staticmethod
    def create_load(name, target_reg, value_source_reg=None):
        return Instruction(name, reads=[value_source_reg] if value_source_reg else [], writes=[target_reg], is_branch=False)

    @staticmethod
    def create_alu(name, reads, writes):
        return Instruction(name, reads=reads, writes=writes, is_branch=False)

    @staticmethod
    def create_branch(name, reads=None, writes=None):
        return Instruction(name, reads=reads or [], writes=writes or [], is_branch=True)

class Pipeline:
    def __init__(self, stages, instructions, branch_predictor):
        self.stages = stages  # e.g. ["IF","ID","EX","MEM","WB"]
        self.instr_queue = instructions[:] 
        self.pipeline_regs = [None] * len(stages) 
        self.cycle = 0
        self.observers = []
        self.stalls = 0
        self.flushed = 0
        self.branch_predictor = branch_predictor

        self.timeline = [] 

    def attach(self, obs):
        self.observers.append(obs)

    def notify(self):
        for o in self.observers:
            o.update_view()

    def detect_data_hazard(self, instr, earlier_instr):
        if earlier_instr is None: 
            return False
        for r in instr.reads:
            if r and r in earlier_instr.writes:
                return True
        return False

    def step(self): 
        self.cycle += 1
        cycle_events = {}  
        stages_count = len(self.stages)
        stall_needed = False
        id_instr = self.pipeline_regs[1]
        ex_instr = self.pipeline_regs[2]
        if id_instr and ex_instr:
            if self.detect_data_hazard(id_instr[1], ex_instr[1]):
                stall_needed = True
        if id_instr and id_instr[1].is_branch and id_instr[1].predicted_taken is None:
            id_instr[1].predicted_taken = self.branch_predictor.predict(id_instr[1])
            cycle_events[(id_instr[0], 1)] = f"PRED:{'T' if id_instr[1].predicted_taken else 'N'}"


        if stall_needed:
            self.stalls += 1
            cycle_events[("stall", self.cycle)] = "DATA_STALL"
            self.pipeline_regs[4] = self.pipeline_regs[3]
            self.pipeline_regs[3] = self.pipeline_regs[2]
            self.pipeline_regs[2] = None
        else: 
            self.pipeline_regs[4] = self.pipeline_regs[3]
            self.pipeline_regs[3] = self.pipeline_regs[2]
            self.pipeline_regs[2] = self.pipeline_regs[1]
            self.pipeline_regs[1] = self.pipeline_regs[0]

            if self.instr_queue:
                instr = self.instr_queue.pop(0)
                instr_index = len(self.timeline) 
                self.pipeline_regs[0] = (instr_index, instr)
            else:
                self.pipeline_regs[0] = None
        ex_entry = self.pipeline_regs[2]
        if ex_entry and ex_entry[1].is_branch:
            instr_obj = ex_entry[1]
            instr_obj.actual_taken = self.branch_predictor.actual_outcome(instr_obj)
            if instr_obj.predicted_taken != instr_obj.actual_taken: 
                self.flushed += 1
                cycle_events[("branch", ex_entry[0])] = "MISPREDICT"
                if self.pipeline_regs[0]:
                    self.pipeline_regs[0] = None
                if self.pipeline_regs[1]:
                    self.pipeline_regs[1] = None
            else:
                cycle_events[("branch", ex_entry[0])] = "PRED_CORRECT"

        for s_idx in range(stages_count):
            entry = self.pipeline_regs[s_idx]
            if entry is None:
                status = ""
            else:
                idx, ins = entry
                status = ins.name
            cycle_events[(s_idx, )] = status

        self.timeline.append(cycle_events)
        self.notify()

    def is_done(self):
        if any(self.pipeline_regs):
            return False
        if self.instr_queue:
            return False
        return True

class SimpleBranchPredictor:
    def __init__(self, mode="static_not_taken"):
        self.mode = mode

    def predict(self, instr):
        if self.mode == "static_taken":
            return True
        if self.mode == "static_not_taken":
            return False
        # default random
        return random.choice([True, False])

    def actual_outcome(self, instr):
        return random.random() < 0.4   


class PipelineView:
    def __init__(self, root, pipeline: Pipeline):
        self.root = root
        self.pipeline = pipeline
        pipeline.attach(self)

        self.root.title("Pipeline Simulation - Branch & Data Hazard Visualization")
        self.root.geometry("980x520")
        self.root.config(bg="#1f2937")

        self.stages = pipeline.stages

        container = tk.Frame(root, bg="#1f2937")
        container.pack(fill="both", expand=True, pady=10)

        self.canvas = tk.Canvas(container, bg="#0f172a", width=940, height=360)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame = tk.Frame(self.canvas, bg="#0f172a")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        ctrl = tk.Frame(root, bg="#1f2937")
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Step Cycle", command=self.step).pack(side="left", padx=6, pady=6)
        ttk.Button(ctrl, text="Run until done", command=self.run_all).pack(side="left", padx=6, pady=6)

        self.info = tk.Label(ctrl, text="Cycle: 0   Stalls: 0   Flushes: 0", fg="white", bg="#1f2937", font=("Arial", 11))
        self.info.pack(side="right", padx=10)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.draw_headers()
        self.boxes = []

    def draw_headers(self):
        self.canvas.delete("all")
        x0, y0 = 120, 20
        box_w, box_h = 140, 36
        for j, st in enumerate(self.stages):
            x = x0 + j * box_w
            self.canvas.create_rectangle(x, y0, x + box_w, y0 + box_h, fill="#334155", outline="white")
            self.canvas.create_text(x + box_w/2, y0 + box_h/2, text=st, fill="white", font=("Arial", 11, "bold"))

    def update_view(self):
        self.draw_headers()
        x0, y0 = 120, 70
        box_w, box_h = 140, 46
        cycles = len(self.pipeline.timeline)

        for i, cycle_events in enumerate(self.pipeline.timeline):
            self.canvas.create_text(40, y0 + i * box_h + box_h/2, text=f"C{ i+1 }", fill="white")
            for j in range(len(self.stages)):
                x = x0 + j * box_w
                y = y0 + i * box_h
                key = (j,)
                status = cycle_events.get(key, "")
                fill = "#0b1220"
                text = status if status else ""
                if text == "":
                    fill = "#0b1220"
                else:
                    fill = "#064e3b"
                self.canvas.create_rectangle(x, y, x + box_w, y + box_h, fill=fill, outline="#334155")
                self.canvas.create_text(x + box_w/2, y + box_h/2, text=text, fill="white", font=("Arial", 10))

        self.info.config(text=f"Cycle: {self.pipeline.cycle}   Stalls: {self.pipeline.stalls}   Flushes: {self.pipeline.flushed}")
        self.root.update_idletasks()

    def step(self):
        if not self.pipeline.is_done():
            self.pipeline.step()
        else:
            self.pipeline.step()

    def run_all(self):
        def runloop():
            if not self.pipeline.is_done():
                self.pipeline.step()
                self.root.after(600, runloop)
        runloop()

    def _on_mousewheel(self, event):
        """Handles mouse scroll inside canvas"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
 
def sample_program():
    f = InstructionFactory
    prog = [
        f.create_load("I1: LOAD R1", target_reg="R1"),
        f.create_alu("I2: ADD R2,R1", reads=["R1"], writes=["R2"]),   
        f.create_branch("I3: BEQ R2,0 -> LABEL", reads=["R2"]),       
        f.create_alu("I4: ADD R3,R5", reads=["R5"], writes=["R3"]),   
        f.create_alu("I5: SUB R6,R7", reads=["R7"], writes=["R6"]),   
    ]
    return prog


def main():
    root = tk.Tk()
    stages = ["IF","ID","EX","MEM","WB"]
    predictor = SimpleBranchPredictor(mode="static_not_taken")  
    pipeline = Pipeline(stages, sample_program(), predictor)
    view = PipelineView(root, pipeline)
    root.mainloop()

if __name__ == "__main__":
    main()
