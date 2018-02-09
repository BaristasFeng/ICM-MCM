#!/usr/bin/python
# _*_ encoding=utf-8
class DescriptionState:
    #State information:private
    MSN=""
    State_Code=""
    Year=0
    Data=0.0

    def __init__(self,M,S,Y,D):
        super().__init__()
        self.MSN=M
        self.State_Code=S
        self.Year=Y
        self.Data=D

    def __str__(self) -> str:
        return "MSN: %s  StateCode: %s Year: %d Data: %f" % (self.MSN, self.State_Code,self.Year,self.Data)


    def MSN(self):
        return self.MSN

    def State_Code(self):
        return self.State_Code

    def Year(self):
        return self.Year

    def get_Data(self):
        return self.Data





