import shutil
import sys

class TerminalProgressBar:
    def __init__(self,total,label="",width=32,stream=None):
        self.total=max(1,int(total))
        self.label=str(label or "").strip()
        self.width=max(10,int(width))
        self.stream=stream or sys.stdout
        self.current=0
        self.last_line=""
        self.last_message=""
        self.closed=False

    def set_total(self,total):
        if self.closed:
            return
        self.total=max(1,int(total))
        if self.current>self.total:
            self.current=self.total

    def _render(self,message=""):
        filled=int(self.width*self.current/self.total)
        empty=self.width-filled
        bar="#"*filled+"-"*empty
        pct=100.0*self.current/self.total
        prefix=f"{self.label} " if self.label else ""
        suffix=f" | {message}" if message else ""
        line=f"{prefix}[{bar}] {self.current}/{self.total} {pct:6.2f}%{suffix}"
        cols=shutil.get_terminal_size((120,20)).columns
        if len(line)>cols:
            line=line[:max(1,cols)]
        return line

    def clear(self):
        if self.closed:
            return
        if not self.last_line:
            return
        self.stream.write("\r"+" "*len(self.last_line)+"\r")
        self.stream.flush()
        self.last_line=""

    def update(self,current=None,step=1,message=""):
        if self.closed:
            return
        if current is None:
            self.current=min(self.total,self.current+int(step))
        else:
            self.current=max(0,min(self.total,int(current)))

        prev_len=len(self.last_line)
        line=self._render(message)
        if len(line)<prev_len:
            line=line+" "*(prev_len-len(line))

        self.stream.write("\r"+line)
        self.stream.flush()
        self.last_line=line
        self.last_message=message

    def finish(self,message="complete"):
        if self.closed:
            return
        self.current=self.total
        prev_len=len(self.last_line)
        line=self._render(message)
        if len(line)<prev_len:
            line=line+" "*(prev_len-len(line))
        self.stream.write("\r"+line)
        self.stream.write("\n")
        self.stream.flush()
        self.last_line=""
        self.last_message=message
        self.closed=True