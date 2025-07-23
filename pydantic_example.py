from pydantic import BaseModel , Field
from typing import Optional
from pydantic import EmailStr

class Students(BaseModel):
    name: str = 'Abdul'
    age: int = 12
    grade: str = 'A'
    Roll_no : Optional[str] = None
    email : EmailStr = "xyz@gmail.com"
    cgpa : float = Field(gt=0,lt=4 ,default=3)
    
new_student = {'Roll_no':"21SW039",'email':"abdul.razzaq@gmail.com"}
# new_student = {'cgpa':3.5}

# student1 = Students(name="John", age="20", grade="A")
student1 = Students(**new_student)

# print(student1)
# print(student1.age)
# print(student1.Roll_no)
# print(student1.email)
# print(student1.cgpa)


student_dict = dict(student1)
print(student_dict['Roll_no'])