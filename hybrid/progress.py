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
        self.closed=False

    def _render(self,message=""):
        filled=int(self.width*self.current/self.total)
        empty=self.width-filled
        bar="#"*filled+"-"*empty
        pct=100.0*self.current/self.total
        prefix=f"{self.label} " if self.label else ""
        suffix=f" | {message}" if message else ""
        line=f"\r{prefix}[{bar}] {self.current}/{self.total} {pct:6.2f}%{suffix}"
        cols=shutil.get_terminal_size((120,20)).columns
        if len(line)>cols:
            line=line[:max(1,cols-1)]
        if len(line)<len(self.last_line):
            line=line+" "*(len(self.last_line)-len(line))
        return line

    def update(self,current=None,step=1,message=""):
        if self.closed:
            return
        if current is None:
            self.current=min(self.total,self.current+int(step))
        else:
            self.current=max(0,min(self.total,int(current)))
        line=self._render(message)
        self.stream.write(line)
        self.stream.flush()
        self.last_line=line

    def finish(self,message="complete"):
        if self.closed:
            return
        self.current=self.total
        line=self._render(message)
        self.stream.write(line)
        self.stream.write("\n")
        self.stream.flush()
        self.last_line=""
        self.closed=True