Йц
Мя
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

√
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8ч─
Ц
Adam/prediction_output_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/v
П
3Adam/prediction_output_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/v*
_output_shapes
:*
dtype0
Ю
!Adam/prediction_output_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/v
Ч
5Adam/prediction_output_0/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/v
y
(Adam/dense_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_83/kernel/v
Б
*Adam/dense_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/v
y
(Adam/dense_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/v
Б
*Adam/dense_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_81/bias/v
y
(Adam/dense_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T*'
shared_nameAdam/dense_81/kernel/v
Б
*Adam/dense_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/v*
_output_shapes

:T*
dtype0
Д
Adam/conv2d_137/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_137/bias/v
}
*Adam/conv2d_137/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_137/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_137/kernel/v
Н
,Adam/conv2d_137/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_136/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_136/bias/v
}
*Adam/conv2d_136/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_136/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_136/kernel/v
Н
,Adam/conv2d_136/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/kernel/v*&
_output_shapes
:	*
dtype0
Д
Adam/conv2d_140/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_140/bias/v
}
*Adam/conv2d_140/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_140/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_140/kernel/v
Н
,Adam/conv2d_140/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/kernel/v*&
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_27/beta/v
Х
6Adam/batch_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_27/gamma/v
Ч
7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/v*
_output_shapes
:*
dtype0
Д
Adam/conv2d_139/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_139/bias/v
}
*Adam/conv2d_139/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_139/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_139/kernel/v
Н
,Adam/conv2d_139/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_135/bias/v
}
*Adam/conv2d_135/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_135/kernel/v
Н
,Adam/conv2d_135/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_138/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_138/bias/v
}
*Adam/conv2d_138/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_138/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_138/kernel/v
Н
,Adam/conv2d_138/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/kernel/v*&
_output_shapes
:		*
dtype0
Ц
Adam/prediction_output_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/m
П
3Adam/prediction_output_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/m*
_output_shapes
:*
dtype0
Ю
!Adam/prediction_output_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/m
Ч
5Adam/prediction_output_0/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/m
y
(Adam/dense_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_83/kernel/m
Б
*Adam/dense_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/m
y
(Adam/dense_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/m
Б
*Adam/dense_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_81/bias/m
y
(Adam/dense_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T*'
shared_nameAdam/dense_81/kernel/m
Б
*Adam/dense_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/m*
_output_shapes

:T*
dtype0
Д
Adam/conv2d_137/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_137/bias/m
}
*Adam/conv2d_137/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_137/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_137/kernel/m
Н
,Adam/conv2d_137/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_137/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_136/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_136/bias/m
}
*Adam/conv2d_136/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_136/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_136/kernel/m
Н
,Adam/conv2d_136/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_136/kernel/m*&
_output_shapes
:	*
dtype0
Д
Adam/conv2d_140/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_140/bias/m
}
*Adam/conv2d_140/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_140/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_140/kernel/m
Н
,Adam/conv2d_140/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_140/kernel/m*&
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_27/beta/m
Х
6Adam/batch_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_27/gamma/m
Ч
7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/m*
_output_shapes
:*
dtype0
Д
Adam/conv2d_139/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_139/bias/m
}
*Adam/conv2d_139/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_139/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_139/kernel/m
Н
,Adam/conv2d_139/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_139/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_135/bias/m
}
*Adam/conv2d_135/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_135/kernel/m
Н
,Adam/conv2d_135/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_135/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_138/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_138/bias/m
}
*Adam/conv2d_138/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_138/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_138/kernel/m
Н
,Adam/conv2d_138/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_138/kernel/m*&
_output_shapes
:		*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
И
prediction_output_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameprediction_output_0/bias
Б
,prediction_output_0/bias/Read/ReadVariableOpReadVariableOpprediction_output_0/bias*
_output_shapes
:*
dtype0
Р
prediction_output_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameprediction_output_0/kernel
Й
.prediction_output_0/kernel/Read/ReadVariableOpReadVariableOpprediction_output_0/kernel*
_output_shapes

:*
dtype0
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
:*
dtype0
z
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_83/kernel
s
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes

:*
dtype0
r
dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_82/bias
k
!dense_82/bias/Read/ReadVariableOpReadVariableOpdense_82/bias*
_output_shapes
:*
dtype0
z
dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_82/kernel
s
#dense_82/kernel/Read/ReadVariableOpReadVariableOpdense_82/kernel*
_output_shapes

:*
dtype0
r
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_81/bias
k
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
_output_shapes
:*
dtype0
z
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T* 
shared_namedense_81/kernel
s
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel*
_output_shapes

:T*
dtype0
v
conv2d_137/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_137/bias
o
#conv2d_137/bias/Read/ReadVariableOpReadVariableOpconv2d_137/bias*
_output_shapes
:*
dtype0
Ж
conv2d_137/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_137/kernel

%conv2d_137/kernel/Read/ReadVariableOpReadVariableOpconv2d_137/kernel*&
_output_shapes
:*
dtype0
v
conv2d_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_136/bias
o
#conv2d_136/bias/Read/ReadVariableOpReadVariableOpconv2d_136/bias*
_output_shapes
:*
dtype0
Ж
conv2d_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameconv2d_136/kernel

%conv2d_136/kernel/Read/ReadVariableOpReadVariableOpconv2d_136/kernel*&
_output_shapes
:	*
dtype0
v
conv2d_140/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_140/bias
o
#conv2d_140/bias/Read/ReadVariableOpReadVariableOpconv2d_140/bias*
_output_shapes
:*
dtype0
Ж
conv2d_140/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_140/kernel

%conv2d_140/kernel/Read/ReadVariableOpReadVariableOpconv2d_140/kernel*&
_output_shapes
:*
dtype0
д
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_27/moving_variance
Э
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_27/moving_mean
Х
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:*
dtype0
О
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_27/beta
З
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:*
dtype0
Р
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_27/gamma
Й
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:*
dtype0
v
conv2d_139/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_139/bias
o
#conv2d_139/bias/Read/ReadVariableOpReadVariableOpconv2d_139/bias*
_output_shapes
:*
dtype0
Ж
conv2d_139/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_139/kernel

%conv2d_139/kernel/Read/ReadVariableOpReadVariableOpconv2d_139/kernel*&
_output_shapes
:*
dtype0
v
conv2d_135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_135/bias
o
#conv2d_135/bias/Read/ReadVariableOpReadVariableOpconv2d_135/bias*
_output_shapes
:*
dtype0
Ж
conv2d_135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_135/kernel

%conv2d_135/kernel/Read/ReadVariableOpReadVariableOpconv2d_135/kernel*&
_output_shapes
:*
dtype0
v
conv2d_138/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_138/bias
o
#conv2d_138/bias/Read/ReadVariableOpReadVariableOpconv2d_138/bias*
_output_shapes
:*
dtype0
Ж
conv2d_138/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*"
shared_nameconv2d_138/kernel

%conv2d_138/kernel/Read/ReadVariableOpReadVariableOpconv2d_138/kernel*&
_output_shapes
:		*
dtype0
Л
serving_default_input_46Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
Л
serving_default_input_47Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_46serving_default_input_47conv2d_138/kernelconv2d_138/biasconv2d_135/kernelconv2d_135/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_139/kernelconv2d_139/biasconv2d_136/kernelconv2d_136/biasconv2d_140/kernelconv2d_140/biasconv2d_137/kernelconv2d_137/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biasprediction_output_0/kernelprediction_output_0/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_1349024

NoOpNoOp
їа
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*па
valueдаBаа BШа
▌
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
* 
е
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator* 
╚
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
╚
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
╒
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance*
╚
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op*
╚
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op*
О
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
╚
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op*
О
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
О
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
О
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
к
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias*
о
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
Кkernel
	Лbias*
о
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkernel
	Уbias*
о
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъkernel
	Ыbias*
┬
"0
#1
22
33
;4
<5
E6
F7
G8
H9
O10
P11
X12
Y13
g14
h15
В16
Г17
К18
Л19
Т20
У21
Ъ22
Ы23*
▓
"0
#1
22
33
;4
<5
E6
F7
O8
P9
X10
Y11
g12
h13
В14
Г15
К16
Л17
Т18
У19
Ъ20
Ы21*
* 
╡
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
бtrace_0
вtrace_1
гtrace_2
дtrace_3* 
:
еtrace_0
жtrace_1
зtrace_2
иtrace_3* 
* 
С
	йiter
кbeta_1
лbeta_2

мdecay
нlearning_rate"mи#mй2mк3mл;mм<mнEmоFmпOm░Pm▒Xm▓Ym│gm┤hm╡	Вm╢	Гm╖	Кm╕	Лm╣	Тm║	Уm╗	Ъm╝	Ыm╜"v╛#v┐2v└3v┴;v┬<v├Ev─Fv┼Ov╞Pv╟Xv╚Yv╔gv╩hv╦	Вv╠	Гv═	Кv╬	Лv╧	Тv╨	Уv╤	Ъv╥	Ыv╙*

оserving_default* 

"0
#1*

"0
#1*
* 
Ш
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

┤trace_0* 

╡trace_0* 
a[
VARIABLE_VALUEconv2d_138/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_138/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

╗trace_0
╝trace_1* 

╜trace_0
╛trace_1* 
* 

20
31*

20
31*
* 
Ш
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
a[
VARIABLE_VALUEconv2d_135/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_135/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

;0
<1*

;0
<1*
* 
Ш
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

╦trace_0* 

╠trace_0* 
a[
VARIABLE_VALUEconv2d_139/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_139/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
E0
F1
G2
H3*

E0
F1*
* 
Ш
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

╥trace_0
╙trace_1* 

╘trace_0
╒trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_27/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_27/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_27/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_27/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

O0
P1*

O0
P1*
* 
Ш
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

█trace_0* 

▄trace_0* 
a[
VARIABLE_VALUEconv2d_140/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_140/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

X0
Y1*

X0
Y1*
* 
Ш
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
a[
VARIABLE_VALUEconv2d_136/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_136/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 

g0
h1*

g0
h1*
* 
Ш
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
a[
VARIABLE_VALUEconv2d_137/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_137/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

ўtrace_0* 

°trace_0* 
* 
* 
* 
Ц
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

■trace_0* 

 trace_0* 
* 
* 
* 
Ц
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 

В0
Г1*

В0
Г1*
* 
Ы
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
_Y
VARIABLE_VALUEdense_81/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_81/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

К0
Л1*

К0
Л1*
* 
Ю
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
_Y
VARIABLE_VALUEdense_82/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_82/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Т0
У1*

Т0
У1*
* 
Ю
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

Ъtrace_0* 

Ыtrace_0* 
_Y
VARIABLE_VALUEdense_83/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_83/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ъ0
Ы1*

Ъ0
Ы1*
* 
Ю
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
ke
VARIABLE_VALUEprediction_output_0/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEprediction_output_0/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*
К
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

г0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
д	variables
е	keras_api

жtotal

зcount*

ж0
з1*

д	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_138/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_138/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_135/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_135/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_139/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_139/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_27/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_140/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_140/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_136/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_136/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_137/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_137/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_81/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_81/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_82/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_83/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/prediction_output_0/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_138/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_138/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_135/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_135/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_139/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_139/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_27/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_140/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_140/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_136/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_136/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_137/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_137/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_81/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_81/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_82/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_83/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/prediction_output_0/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ф
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_138/kernel/Read/ReadVariableOp#conv2d_138/bias/Read/ReadVariableOp%conv2d_135/kernel/Read/ReadVariableOp#conv2d_135/bias/Read/ReadVariableOp%conv2d_139/kernel/Read/ReadVariableOp#conv2d_139/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp%conv2d_140/kernel/Read/ReadVariableOp#conv2d_140/bias/Read/ReadVariableOp%conv2d_136/kernel/Read/ReadVariableOp#conv2d_136/bias/Read/ReadVariableOp%conv2d_137/kernel/Read/ReadVariableOp#conv2d_137/bias/Read/ReadVariableOp#dense_81/kernel/Read/ReadVariableOp!dense_81/bias/Read/ReadVariableOp#dense_82/kernel/Read/ReadVariableOp!dense_82/bias/Read/ReadVariableOp#dense_83/kernel/Read/ReadVariableOp!dense_83/bias/Read/ReadVariableOp.prediction_output_0/kernel/Read/ReadVariableOp,prediction_output_0/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_138/kernel/m/Read/ReadVariableOp*Adam/conv2d_138/bias/m/Read/ReadVariableOp,Adam/conv2d_135/kernel/m/Read/ReadVariableOp*Adam/conv2d_135/bias/m/Read/ReadVariableOp,Adam/conv2d_139/kernel/m/Read/ReadVariableOp*Adam/conv2d_139/bias/m/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_27/beta/m/Read/ReadVariableOp,Adam/conv2d_140/kernel/m/Read/ReadVariableOp*Adam/conv2d_140/bias/m/Read/ReadVariableOp,Adam/conv2d_136/kernel/m/Read/ReadVariableOp*Adam/conv2d_136/bias/m/Read/ReadVariableOp,Adam/conv2d_137/kernel/m/Read/ReadVariableOp*Adam/conv2d_137/bias/m/Read/ReadVariableOp*Adam/dense_81/kernel/m/Read/ReadVariableOp(Adam/dense_81/bias/m/Read/ReadVariableOp*Adam/dense_82/kernel/m/Read/ReadVariableOp(Adam/dense_82/bias/m/Read/ReadVariableOp*Adam/dense_83/kernel/m/Read/ReadVariableOp(Adam/dense_83/bias/m/Read/ReadVariableOp5Adam/prediction_output_0/kernel/m/Read/ReadVariableOp3Adam/prediction_output_0/bias/m/Read/ReadVariableOp,Adam/conv2d_138/kernel/v/Read/ReadVariableOp*Adam/conv2d_138/bias/v/Read/ReadVariableOp,Adam/conv2d_135/kernel/v/Read/ReadVariableOp*Adam/conv2d_135/bias/v/Read/ReadVariableOp,Adam/conv2d_139/kernel/v/Read/ReadVariableOp*Adam/conv2d_139/bias/v/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_27/beta/v/Read/ReadVariableOp,Adam/conv2d_140/kernel/v/Read/ReadVariableOp*Adam/conv2d_140/bias/v/Read/ReadVariableOp,Adam/conv2d_136/kernel/v/Read/ReadVariableOp*Adam/conv2d_136/bias/v/Read/ReadVariableOp,Adam/conv2d_137/kernel/v/Read/ReadVariableOp*Adam/conv2d_137/bias/v/Read/ReadVariableOp*Adam/dense_81/kernel/v/Read/ReadVariableOp(Adam/dense_81/bias/v/Read/ReadVariableOp*Adam/dense_82/kernel/v/Read/ReadVariableOp(Adam/dense_82/bias/v/Read/ReadVariableOp*Adam/dense_83/kernel/v/Read/ReadVariableOp(Adam/dense_83/bias/v/Read/ReadVariableOp5Adam/prediction_output_0/kernel/v/Read/ReadVariableOp3Adam/prediction_output_0/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_save_1349911
│
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_138/kernelconv2d_138/biasconv2d_135/kernelconv2d_135/biasconv2d_139/kernelconv2d_139/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_140/kernelconv2d_140/biasconv2d_136/kernelconv2d_136/biasconv2d_137/kernelconv2d_137/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biasprediction_output_0/kernelprediction_output_0/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_138/kernel/mAdam/conv2d_138/bias/mAdam/conv2d_135/kernel/mAdam/conv2d_135/bias/mAdam/conv2d_139/kernel/mAdam/conv2d_139/bias/m#Adam/batch_normalization_27/gamma/m"Adam/batch_normalization_27/beta/mAdam/conv2d_140/kernel/mAdam/conv2d_140/bias/mAdam/conv2d_136/kernel/mAdam/conv2d_136/bias/mAdam/conv2d_137/kernel/mAdam/conv2d_137/bias/mAdam/dense_81/kernel/mAdam/dense_81/bias/mAdam/dense_82/kernel/mAdam/dense_82/bias/mAdam/dense_83/kernel/mAdam/dense_83/bias/m!Adam/prediction_output_0/kernel/mAdam/prediction_output_0/bias/mAdam/conv2d_138/kernel/vAdam/conv2d_138/bias/vAdam/conv2d_135/kernel/vAdam/conv2d_135/bias/vAdam/conv2d_139/kernel/vAdam/conv2d_139/bias/v#Adam/batch_normalization_27/gamma/v"Adam/batch_normalization_27/beta/vAdam/conv2d_140/kernel/vAdam/conv2d_140/bias/vAdam/conv2d_136/kernel/vAdam/conv2d_136/bias/vAdam/conv2d_137/kernel/vAdam/conv2d_137/bias/vAdam/dense_81/kernel/vAdam/dense_81/bias/vAdam/dense_82/kernel/vAdam/dense_82/bias/vAdam/dense_83/kernel/vAdam/dense_83/bias/v!Adam/prediction_output_0/kernel/vAdam/prediction_output_0/bias/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__traced_restore_1350146т╤
ТN
┘
H__inference_joint_model_layer_call_and_return_conditional_losses_1348423

inputs
inputs_1,
conv2d_138_1348237:		 
conv2d_138_1348239:,
conv2d_135_1348253: 
conv2d_135_1348255:,
batch_normalization_27_1348259:,
batch_normalization_27_1348261:,
batch_normalization_27_1348263:,
batch_normalization_27_1348265:,
conv2d_139_1348279: 
conv2d_139_1348281:,
conv2d_136_1348295:	 
conv2d_136_1348297:,
conv2d_140_1348311: 
conv2d_140_1348313:,
conv2d_137_1348327: 
conv2d_137_1348329:"
dense_81_1348369:T
dense_81_1348371:"
dense_82_1348385:
dense_82_1348387:"
dense_83_1348401:
dense_83_1348403:-
prediction_output_0_1348417:)
prediction_output_0_1348419:
identityИв.batch_normalization_27/StatefulPartitionedCallв"conv2d_135/StatefulPartitionedCallв"conv2d_136/StatefulPartitionedCallв"conv2d_137/StatefulPartitionedCallв"conv2d_138/StatefulPartitionedCallв"conv2d_139/StatefulPartitionedCallв"conv2d_140/StatefulPartitionedCallв dense_81/StatefulPartitionedCallв dense_82/StatefulPartitionedCallв dense_83/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallИ
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_138_1348237conv2d_138_1348239*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1348236Ж
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_135_1348253conv2d_135_1348255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1348252Б
$spatial_dropout2d_18/PartitionedCallPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348109Я
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_27_1348259batch_normalization_27_1348261batch_normalization_27_1348263batch_normalization_27_1348265*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348162н
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall-spatial_dropout2d_18/PartitionedCall:output:0conv2d_139_1348279conv2d_139_1348281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1348278╖
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_136_1348295conv2d_136_1348297*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1348294л
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0conv2d_140_1348311conv2d_140_1348313*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1348310л
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0conv2d_137_1348327conv2d_137_1348329*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1348326 
'global_max_pooling2d_18/PartitionedCallPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1348214ъ
flatten_46/PartitionedCallPartitionedCall0global_max_pooling2d_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_1348339х
flatten_45/PartitionedCallPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_1348347Л
concatenate_18/PartitionedCallPartitionedCall#flatten_46/PartitionedCall:output:0#flatten_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1348356Ч
 dense_81/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0dense_81_1348369dense_81_1348371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_1348368Щ
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_1348385dense_82_1348387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1348384Щ
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1348401dense_83_1348403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1348400┼
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0prediction_output_0_1348417prediction_output_0_1348419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1348416Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ь
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
к

А
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1348326

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╨
▒
-__inference_joint_model_layer_call_fn_1348474
input_46
input_47!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:T

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinput_46input_47unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_1348423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_46:YU
/
_output_shapes
:         
"
_user_specified_name
input_47
▄p
Г
H__inference_joint_model_layer_call_and_return_conditional_losses_1349220
inputs_0
inputs_1C
)conv2d_138_conv2d_readvariableop_resource:		8
*conv2d_138_biasadd_readvariableop_resource:C
)conv2d_135_conv2d_readvariableop_resource:8
*conv2d_135_biasadd_readvariableop_resource:<
.batch_normalization_27_readvariableop_resource:>
0batch_normalization_27_readvariableop_1_resource:M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_139_conv2d_readvariableop_resource:8
*conv2d_139_biasadd_readvariableop_resource:C
)conv2d_136_conv2d_readvariableop_resource:	8
*conv2d_136_biasadd_readvariableop_resource:C
)conv2d_140_conv2d_readvariableop_resource:8
*conv2d_140_biasadd_readvariableop_resource:C
)conv2d_137_conv2d_readvariableop_resource:8
*conv2d_137_biasadd_readvariableop_resource:9
'dense_81_matmul_readvariableop_resource:T6
(dense_81_biasadd_readvariableop_resource:9
'dense_82_matmul_readvariableop_resource:6
(dense_82_biasadd_readvariableop_resource:9
'dense_83_matmul_readvariableop_resource:6
(dense_83_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityИв6batch_normalization_27/FusedBatchNormV3/ReadVariableOpв8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_27/ReadVariableOpв'batch_normalization_27/ReadVariableOp_1в!conv2d_135/BiasAdd/ReadVariableOpв conv2d_135/Conv2D/ReadVariableOpв!conv2d_136/BiasAdd/ReadVariableOpв conv2d_136/Conv2D/ReadVariableOpв!conv2d_137/BiasAdd/ReadVariableOpв conv2d_137/Conv2D/ReadVariableOpв!conv2d_138/BiasAdd/ReadVariableOpв conv2d_138/Conv2D/ReadVariableOpв!conv2d_139/BiasAdd/ReadVariableOpв conv2d_139/Conv2D/ReadVariableOpв!conv2d_140/BiasAdd/ReadVariableOpв conv2d_140/Conv2D/ReadVariableOpвdense_81/BiasAdd/ReadVariableOpвdense_81/MatMul/ReadVariableOpвdense_82/BiasAdd/ReadVariableOpвdense_82/MatMul/ReadVariableOpвdense_83/BiasAdd/ReadVariableOpвdense_83/MatMul/ReadVariableOpв*prediction_output_0/BiasAdd/ReadVariableOpв)prediction_output_0/MatMul/ReadVariableOpТ
 conv2d_138/Conv2D/ReadVariableOpReadVariableOp)conv2d_138_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0▒
conv2d_138/Conv2DConv2Dinputs_1(conv2d_138/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

*
paddingSAME*
strides
И
!conv2d_138/BiasAdd/ReadVariableOpReadVariableOp*conv2d_138_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_138/BiasAddBiasAddconv2d_138/Conv2D:output:0)conv2d_138/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

Т
 conv2d_135/Conv2D/ReadVariableOpReadVariableOp)conv2d_135_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▒
conv2d_135/Conv2DConv2Dinputs_0(conv2d_135/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_135/BiasAdd/ReadVariableOpReadVariableOp*conv2d_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_135/BiasAddBiasAddconv2d_135/Conv2D:output:0)conv2d_135/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         А
spatial_dropout2d_18/IdentityIdentityconv2d_138/BiasAdd:output:0*
T0*/
_output_shapes
:         

Р
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype0Ф
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╢
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_135/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( Т
 conv2d_139/Conv2D/ReadVariableOpReadVariableOp)conv2d_139_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╧
conv2d_139/Conv2DConv2D&spatial_dropout2d_18/Identity:output:0(conv2d_139/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_139/BiasAdd/ReadVariableOpReadVariableOp*conv2d_139_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_139/BiasAddBiasAddconv2d_139/Conv2D:output:0)conv2d_139/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Т
 conv2d_136/Conv2D/ReadVariableOpReadVariableOp)conv2d_136_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0╘
conv2d_136/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0(conv2d_136/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_136/BiasAdd/ReadVariableOpReadVariableOp*conv2d_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_136/BiasAddBiasAddconv2d_136/Conv2D:output:0)conv2d_136/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Т
 conv2d_140/Conv2D/ReadVariableOpReadVariableOp)conv2d_140_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
conv2d_140/Conv2DConv2Dconv2d_139/BiasAdd:output:0(conv2d_140/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_140/BiasAdd/ReadVariableOpReadVariableOp*conv2d_140_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_140/BiasAddBiasAddconv2d_140/Conv2D:output:0)conv2d_140/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Т
 conv2d_137/Conv2D/ReadVariableOpReadVariableOp)conv2d_137_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
conv2d_137/Conv2DConv2Dconv2d_136/BiasAdd:output:0(conv2d_137/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_137/BiasAdd/ReadVariableOpReadVariableOp*conv2d_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_137/BiasAddBiasAddconv2d_137/Conv2D:output:0)conv2d_137/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
-global_max_pooling2d_18/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      й
global_max_pooling2d_18/MaxMaxconv2d_140/BiasAdd:output:06global_max_pooling2d_18/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         a
flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_46/ReshapeReshape$global_max_pooling2d_18/Max:output:0flatten_46/Const:output:0*
T0*'
_output_shapes
:         a
flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"    P   З
flatten_45/ReshapeReshapeconv2d_137/BiasAdd:output:0flatten_45/Const:output:0*
T0*'
_output_shapes
:         P\
concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
concatenate_18/concatConcatV2flatten_46/Reshape:output:0flatten_45/Reshape:output:0#concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:         TЖ
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:T*
dtype0У
dense_81/MatMulMatMulconcatenate_18/concat:output:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_82/MatMulMatMuldense_81/BiasAdd:output:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_83/MatMulMatMuldense_82/BiasAdd:output:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0д
prediction_output_0/MatMulMatMuldense_83/BiasAdd:output:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ъ
*prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
prediction_output_0/BiasAddBiasAdd$prediction_output_0/MatMul:product:02prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         s
IdentityIdentity$prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╪
NoOpNoOp7^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1"^conv2d_135/BiasAdd/ReadVariableOp!^conv2d_135/Conv2D/ReadVariableOp"^conv2d_136/BiasAdd/ReadVariableOp!^conv2d_136/Conv2D/ReadVariableOp"^conv2d_137/BiasAdd/ReadVariableOp!^conv2d_137/Conv2D/ReadVariableOp"^conv2d_138/BiasAdd/ReadVariableOp!^conv2d_138/Conv2D/ReadVariableOp"^conv2d_139/BiasAdd/ReadVariableOp!^conv2d_139/Conv2D/ReadVariableOp"^conv2d_140/BiasAdd/ReadVariableOp!^conv2d_140/Conv2D/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12F
!conv2d_135/BiasAdd/ReadVariableOp!conv2d_135/BiasAdd/ReadVariableOp2D
 conv2d_135/Conv2D/ReadVariableOp conv2d_135/Conv2D/ReadVariableOp2F
!conv2d_136/BiasAdd/ReadVariableOp!conv2d_136/BiasAdd/ReadVariableOp2D
 conv2d_136/Conv2D/ReadVariableOp conv2d_136/Conv2D/ReadVariableOp2F
!conv2d_137/BiasAdd/ReadVariableOp!conv2d_137/BiasAdd/ReadVariableOp2D
 conv2d_137/Conv2D/ReadVariableOp conv2d_137/Conv2D/ReadVariableOp2F
!conv2d_138/BiasAdd/ReadVariableOp!conv2d_138/BiasAdd/ReadVariableOp2D
 conv2d_138/Conv2D/ReadVariableOp conv2d_138/Conv2D/ReadVariableOp2F
!conv2d_139/BiasAdd/ReadVariableOp!conv2d_139/BiasAdd/ReadVariableOp2D
 conv2d_139/Conv2D/ReadVariableOp conv2d_139/Conv2D/ReadVariableOp2F
!conv2d_140/BiasAdd/ReadVariableOp!conv2d_140/BiasAdd/ReadVariableOp2D
 conv2d_140/Conv2D/ReadVariableOp conv2d_140/Conv2D/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
╚	
Ў
E__inference_dense_83_layer_call_and_return_conditional_losses_1348400

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╟
Ч
*__inference_dense_81_layer_call_fn_1349595

inputs
unknown:T
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_1348368o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         T: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
Ш
U
9__inference_global_max_pooling2d_18_layer_call_fn_1349526

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1348214i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
А
p
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348137

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?З
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4                                    `
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╓
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:м
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"                  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╖
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"                  А
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"                  М
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4                                    |
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
є
б
,__inference_conv2d_139_layer_call_fn_1349411

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1348278w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         


 
_user_specified_nameinputs
к

А
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1348278

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         


 
_user_specified_nameinputs
ВР
╒
H__inference_joint_model_layer_call_and_return_conditional_losses_1349326
inputs_0
inputs_1C
)conv2d_138_conv2d_readvariableop_resource:		8
*conv2d_138_biasadd_readvariableop_resource:C
)conv2d_135_conv2d_readvariableop_resource:8
*conv2d_135_biasadd_readvariableop_resource:<
.batch_normalization_27_readvariableop_resource:>
0batch_normalization_27_readvariableop_1_resource:M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_139_conv2d_readvariableop_resource:8
*conv2d_139_biasadd_readvariableop_resource:C
)conv2d_136_conv2d_readvariableop_resource:	8
*conv2d_136_biasadd_readvariableop_resource:C
)conv2d_140_conv2d_readvariableop_resource:8
*conv2d_140_biasadd_readvariableop_resource:C
)conv2d_137_conv2d_readvariableop_resource:8
*conv2d_137_biasadd_readvariableop_resource:9
'dense_81_matmul_readvariableop_resource:T6
(dense_81_biasadd_readvariableop_resource:9
'dense_82_matmul_readvariableop_resource:6
(dense_82_biasadd_readvariableop_resource:9
'dense_83_matmul_readvariableop_resource:6
(dense_83_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityИв%batch_normalization_27/AssignNewValueв'batch_normalization_27/AssignNewValue_1в6batch_normalization_27/FusedBatchNormV3/ReadVariableOpв8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_27/ReadVariableOpв'batch_normalization_27/ReadVariableOp_1в!conv2d_135/BiasAdd/ReadVariableOpв conv2d_135/Conv2D/ReadVariableOpв!conv2d_136/BiasAdd/ReadVariableOpв conv2d_136/Conv2D/ReadVariableOpв!conv2d_137/BiasAdd/ReadVariableOpв conv2d_137/Conv2D/ReadVariableOpв!conv2d_138/BiasAdd/ReadVariableOpв conv2d_138/Conv2D/ReadVariableOpв!conv2d_139/BiasAdd/ReadVariableOpв conv2d_139/Conv2D/ReadVariableOpв!conv2d_140/BiasAdd/ReadVariableOpв conv2d_140/Conv2D/ReadVariableOpвdense_81/BiasAdd/ReadVariableOpвdense_81/MatMul/ReadVariableOpвdense_82/BiasAdd/ReadVariableOpвdense_82/MatMul/ReadVariableOpвdense_83/BiasAdd/ReadVariableOpвdense_83/MatMul/ReadVariableOpв*prediction_output_0/BiasAdd/ReadVariableOpв)prediction_output_0/MatMul/ReadVariableOpТ
 conv2d_138/Conv2D/ReadVariableOpReadVariableOp)conv2d_138_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0▒
conv2d_138/Conv2DConv2Dinputs_1(conv2d_138/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

*
paddingSAME*
strides
И
!conv2d_138/BiasAdd/ReadVariableOpReadVariableOp*conv2d_138_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_138/BiasAddBiasAddconv2d_138/Conv2D:output:0)conv2d_138/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

Т
 conv2d_135/Conv2D/ReadVariableOpReadVariableOp)conv2d_135_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▒
conv2d_135/Conv2DConv2Dinputs_0(conv2d_135/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_135/BiasAdd/ReadVariableOpReadVariableOp*conv2d_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_135/BiasAddBiasAddconv2d_135/Conv2D:output:0)conv2d_135/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         e
spatial_dropout2d_18/ShapeShapeconv2d_138/BiasAdd:output:0*
T0*
_output_shapes
:r
(spatial_dropout2d_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spatial_dropout2d_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spatial_dropout2d_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"spatial_dropout2d_18/strided_sliceStridedSlice#spatial_dropout2d_18/Shape:output:01spatial_dropout2d_18/strided_slice/stack:output:03spatial_dropout2d_18/strided_slice/stack_1:output:03spatial_dropout2d_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*spatial_dropout2d_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,spatial_dropout2d_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,spatial_dropout2d_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
$spatial_dropout2d_18/strided_slice_1StridedSlice#spatial_dropout2d_18/Shape:output:03spatial_dropout2d_18/strided_slice_1/stack:output:05spatial_dropout2d_18/strided_slice_1/stack_1:output:05spatial_dropout2d_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
"spatial_dropout2d_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?л
 spatial_dropout2d_18/dropout/MulMulconv2d_138/BiasAdd:output:0+spatial_dropout2d_18/dropout/Const:output:0*
T0*/
_output_shapes
:         

u
3spatial_dropout2d_18/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
3spatial_dropout2d_18/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :┐
1spatial_dropout2d_18/dropout/random_uniform/shapePack+spatial_dropout2d_18/strided_slice:output:0<spatial_dropout2d_18/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_18/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_18/strided_slice_1:output:0*
N*
T0*
_output_shapes
:═
9spatial_dropout2d_18/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_18/dropout/random_uniform/shape:output:0*
T0*/
_output_shapes
:         *
dtype0p
+spatial_dropout2d_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>э
)spatial_dropout2d_18/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_18/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_18/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         б
!spatial_dropout2d_18/dropout/CastCast-spatial_dropout2d_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         ░
"spatial_dropout2d_18/dropout/Mul_1Mul$spatial_dropout2d_18/dropout/Mul:z:0%spatial_dropout2d_18/dropout/Cast:y:0*
T0*/
_output_shapes
:         

Р
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype0Ф
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╢
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╠
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_135/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Т
 conv2d_139/Conv2D/ReadVariableOpReadVariableOp)conv2d_139_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╧
conv2d_139/Conv2DConv2D&spatial_dropout2d_18/dropout/Mul_1:z:0(conv2d_139/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_139/BiasAdd/ReadVariableOpReadVariableOp*conv2d_139_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_139/BiasAddBiasAddconv2d_139/Conv2D:output:0)conv2d_139/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Т
 conv2d_136/Conv2D/ReadVariableOpReadVariableOp)conv2d_136_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0╘
conv2d_136/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0(conv2d_136/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_136/BiasAdd/ReadVariableOpReadVariableOp*conv2d_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_136/BiasAddBiasAddconv2d_136/Conv2D:output:0)conv2d_136/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Т
 conv2d_140/Conv2D/ReadVariableOpReadVariableOp)conv2d_140_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
conv2d_140/Conv2DConv2Dconv2d_139/BiasAdd:output:0(conv2d_140/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_140/BiasAdd/ReadVariableOpReadVariableOp*conv2d_140_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_140/BiasAddBiasAddconv2d_140/Conv2D:output:0)conv2d_140/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Т
 conv2d_137/Conv2D/ReadVariableOpReadVariableOp)conv2d_137_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
conv2d_137/Conv2DConv2Dconv2d_136/BiasAdd:output:0(conv2d_137/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
!conv2d_137/BiasAdd/ReadVariableOpReadVariableOp*conv2d_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_137/BiasAddBiasAddconv2d_137/Conv2D:output:0)conv2d_137/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
-global_max_pooling2d_18/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      й
global_max_pooling2d_18/MaxMaxconv2d_140/BiasAdd:output:06global_max_pooling2d_18/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         a
flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_46/ReshapeReshape$global_max_pooling2d_18/Max:output:0flatten_46/Const:output:0*
T0*'
_output_shapes
:         a
flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"    P   З
flatten_45/ReshapeReshapeconv2d_137/BiasAdd:output:0flatten_45/Const:output:0*
T0*'
_output_shapes
:         P\
concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
concatenate_18/concatConcatV2flatten_46/Reshape:output:0flatten_45/Reshape:output:0#concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:         TЖ
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:T*
dtype0У
dense_81/MatMulMatMulconcatenate_18/concat:output:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_82/MatMulMatMuldense_81/BiasAdd:output:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_83/MatMulMatMuldense_82/BiasAdd:output:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0д
prediction_output_0/MatMulMatMuldense_83/BiasAdd:output:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ъ
*prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
prediction_output_0/BiasAddBiasAdd$prediction_output_0/MatMul:product:02prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         s
IdentityIdentity$prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1"^conv2d_135/BiasAdd/ReadVariableOp!^conv2d_135/Conv2D/ReadVariableOp"^conv2d_136/BiasAdd/ReadVariableOp!^conv2d_136/Conv2D/ReadVariableOp"^conv2d_137/BiasAdd/ReadVariableOp!^conv2d_137/Conv2D/ReadVariableOp"^conv2d_138/BiasAdd/ReadVariableOp!^conv2d_138/Conv2D/ReadVariableOp"^conv2d_139/BiasAdd/ReadVariableOp!^conv2d_139/Conv2D/ReadVariableOp"^conv2d_140/BiasAdd/ReadVariableOp!^conv2d_140/Conv2D/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12F
!conv2d_135/BiasAdd/ReadVariableOp!conv2d_135/BiasAdd/ReadVariableOp2D
 conv2d_135/Conv2D/ReadVariableOp conv2d_135/Conv2D/ReadVariableOp2F
!conv2d_136/BiasAdd/ReadVariableOp!conv2d_136/BiasAdd/ReadVariableOp2D
 conv2d_136/Conv2D/ReadVariableOp conv2d_136/Conv2D/ReadVariableOp2F
!conv2d_137/BiasAdd/ReadVariableOp!conv2d_137/BiasAdd/ReadVariableOp2D
 conv2d_137/Conv2D/ReadVariableOp conv2d_137/Conv2D/ReadVariableOp2F
!conv2d_138/BiasAdd/ReadVariableOp!conv2d_138/BiasAdd/ReadVariableOp2D
 conv2d_138/Conv2D/ReadVariableOp conv2d_138/Conv2D/ReadVariableOp2F
!conv2d_139/BiasAdd/ReadVariableOp!conv2d_139/BiasAdd/ReadVariableOp2D
 conv2d_139/Conv2D/ReadVariableOp conv2d_139/Conv2D/ReadVariableOp2F
!conv2d_140/BiasAdd/ReadVariableOp!conv2d_140/BiasAdd/ReadVariableOp2D
 conv2d_140/Conv2D/ReadVariableOp conv2d_140/Conv2D/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
є
б
,__inference_conv2d_138_layer_call_fn_1349335

inputs!
unknown:		
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1348236w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         

`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╬
▒
-__inference_joint_model_layer_call_fn_1349132
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:T

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_1348719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
╙	
Б
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1349662

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
И
┬
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348193

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
И
┬
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349483

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╚	
Ў
E__inference_dense_83_layer_call_and_return_conditional_losses_1349643

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╬
▒
-__inference_joint_model_layer_call_fn_1348824
input_46
input_47!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:T

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_46input_47unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_1348719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_46:YU
/
_output_shapes
:         
"
_user_specified_name
input_47
є
б
,__inference_conv2d_136_layer_call_fn_1349511

inputs!
unknown:	
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1348294w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
H
,__inference_flatten_46_layer_call_fn_1349556

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_1348339`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│
\
0__inference_concatenate_18_layer_call_fn_1349579
inputs_0
inputs_1
identity╞
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1348356`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         T"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         P:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         P
"
_user_specified_name
inputs/1
ЙИ
Э
"__inference__wrapped_model_1348100
input_46
input_47O
5joint_model_conv2d_138_conv2d_readvariableop_resource:		D
6joint_model_conv2d_138_biasadd_readvariableop_resource:O
5joint_model_conv2d_135_conv2d_readvariableop_resource:D
6joint_model_conv2d_135_biasadd_readvariableop_resource:H
:joint_model_batch_normalization_27_readvariableop_resource:J
<joint_model_batch_normalization_27_readvariableop_1_resource:Y
Kjoint_model_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:[
Mjoint_model_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:O
5joint_model_conv2d_139_conv2d_readvariableop_resource:D
6joint_model_conv2d_139_biasadd_readvariableop_resource:O
5joint_model_conv2d_136_conv2d_readvariableop_resource:	D
6joint_model_conv2d_136_biasadd_readvariableop_resource:O
5joint_model_conv2d_140_conv2d_readvariableop_resource:D
6joint_model_conv2d_140_biasadd_readvariableop_resource:O
5joint_model_conv2d_137_conv2d_readvariableop_resource:D
6joint_model_conv2d_137_biasadd_readvariableop_resource:E
3joint_model_dense_81_matmul_readvariableop_resource:TB
4joint_model_dense_81_biasadd_readvariableop_resource:E
3joint_model_dense_82_matmul_readvariableop_resource:B
4joint_model_dense_82_biasadd_readvariableop_resource:E
3joint_model_dense_83_matmul_readvariableop_resource:B
4joint_model_dense_83_biasadd_readvariableop_resource:P
>joint_model_prediction_output_0_matmul_readvariableop_resource:M
?joint_model_prediction_output_0_biasadd_readvariableop_resource:
identityИвBjoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOpвDjoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1в1joint_model/batch_normalization_27/ReadVariableOpв3joint_model/batch_normalization_27/ReadVariableOp_1в-joint_model/conv2d_135/BiasAdd/ReadVariableOpв,joint_model/conv2d_135/Conv2D/ReadVariableOpв-joint_model/conv2d_136/BiasAdd/ReadVariableOpв,joint_model/conv2d_136/Conv2D/ReadVariableOpв-joint_model/conv2d_137/BiasAdd/ReadVariableOpв,joint_model/conv2d_137/Conv2D/ReadVariableOpв-joint_model/conv2d_138/BiasAdd/ReadVariableOpв,joint_model/conv2d_138/Conv2D/ReadVariableOpв-joint_model/conv2d_139/BiasAdd/ReadVariableOpв,joint_model/conv2d_139/Conv2D/ReadVariableOpв-joint_model/conv2d_140/BiasAdd/ReadVariableOpв,joint_model/conv2d_140/Conv2D/ReadVariableOpв+joint_model/dense_81/BiasAdd/ReadVariableOpв*joint_model/dense_81/MatMul/ReadVariableOpв+joint_model/dense_82/BiasAdd/ReadVariableOpв*joint_model/dense_82/MatMul/ReadVariableOpв+joint_model/dense_83/BiasAdd/ReadVariableOpв*joint_model/dense_83/MatMul/ReadVariableOpв6joint_model/prediction_output_0/BiasAdd/ReadVariableOpв5joint_model/prediction_output_0/MatMul/ReadVariableOpк
,joint_model/conv2d_138/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_138_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0╔
joint_model/conv2d_138/Conv2DConv2Dinput_474joint_model/conv2d_138/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

*
paddingSAME*
strides
а
-joint_model/conv2d_138/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_138_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
joint_model/conv2d_138/BiasAddBiasAdd&joint_model/conv2d_138/Conv2D:output:05joint_model/conv2d_138/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

к
,joint_model/conv2d_135/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_135_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╔
joint_model/conv2d_135/Conv2DConv2Dinput_464joint_model/conv2d_135/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
а
-joint_model/conv2d_135/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
joint_model/conv2d_135/BiasAddBiasAdd&joint_model/conv2d_135/Conv2D:output:05joint_model/conv2d_135/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Ш
)joint_model/spatial_dropout2d_18/IdentityIdentity'joint_model/conv2d_138/BiasAdd:output:0*
T0*/
_output_shapes
:         

и
1joint_model/batch_normalization_27/ReadVariableOpReadVariableOp:joint_model_batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype0м
3joint_model/batch_normalization_27/ReadVariableOp_1ReadVariableOp<joint_model_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bjoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpKjoint_model_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Djoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMjoint_model_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ж
3joint_model/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3'joint_model/conv2d_135/BiasAdd:output:09joint_model/batch_normalization_27/ReadVariableOp:value:0;joint_model/batch_normalization_27/ReadVariableOp_1:value:0Jjoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Ljoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( к
,joint_model/conv2d_139/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_139_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0є
joint_model/conv2d_139/Conv2DConv2D2joint_model/spatial_dropout2d_18/Identity:output:04joint_model/conv2d_139/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
а
-joint_model/conv2d_139/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_139_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
joint_model/conv2d_139/BiasAddBiasAdd&joint_model/conv2d_139/Conv2D:output:05joint_model/conv2d_139/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         к
,joint_model/conv2d_136/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_136_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0°
joint_model/conv2d_136/Conv2DConv2D7joint_model/batch_normalization_27/FusedBatchNormV3:y:04joint_model/conv2d_136/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
а
-joint_model/conv2d_136/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
joint_model/conv2d_136/BiasAddBiasAdd&joint_model/conv2d_136/Conv2D:output:05joint_model/conv2d_136/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         к
,joint_model/conv2d_140/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_140_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ш
joint_model/conv2d_140/Conv2DConv2D'joint_model/conv2d_139/BiasAdd:output:04joint_model/conv2d_140/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
а
-joint_model/conv2d_140/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_140_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
joint_model/conv2d_140/BiasAddBiasAdd&joint_model/conv2d_140/Conv2D:output:05joint_model/conv2d_140/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         к
,joint_model/conv2d_137/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_137_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ш
joint_model/conv2d_137/Conv2DConv2D'joint_model/conv2d_136/BiasAdd:output:04joint_model/conv2d_137/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
а
-joint_model/conv2d_137/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
joint_model/conv2d_137/BiasAddBiasAdd&joint_model/conv2d_137/Conv2D:output:05joint_model/conv2d_137/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         К
9joint_model/global_max_pooling2d_18/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ═
'joint_model/global_max_pooling2d_18/MaxMax'joint_model/conv2d_140/BiasAdd:output:0Bjoint_model/global_max_pooling2d_18/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         m
joint_model/flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ┤
joint_model/flatten_46/ReshapeReshape0joint_model/global_max_pooling2d_18/Max:output:0%joint_model/flatten_46/Const:output:0*
T0*'
_output_shapes
:         m
joint_model/flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"    P   л
joint_model/flatten_45/ReshapeReshape'joint_model/conv2d_137/BiasAdd:output:0%joint_model/flatten_45/Const:output:0*
T0*'
_output_shapes
:         Ph
&joint_model/concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ы
!joint_model/concatenate_18/concatConcatV2'joint_model/flatten_46/Reshape:output:0'joint_model/flatten_45/Reshape:output:0/joint_model/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:         TЮ
*joint_model/dense_81/MatMul/ReadVariableOpReadVariableOp3joint_model_dense_81_matmul_readvariableop_resource*
_output_shapes

:T*
dtype0╖
joint_model/dense_81/MatMulMatMul*joint_model/concatenate_18/concat:output:02joint_model/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+joint_model/dense_81/BiasAdd/ReadVariableOpReadVariableOp4joint_model_dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
joint_model/dense_81/BiasAddBiasAdd%joint_model/dense_81/MatMul:product:03joint_model/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
*joint_model/dense_82/MatMul/ReadVariableOpReadVariableOp3joint_model_dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0▓
joint_model/dense_82/MatMulMatMul%joint_model/dense_81/BiasAdd:output:02joint_model/dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+joint_model/dense_82/BiasAdd/ReadVariableOpReadVariableOp4joint_model_dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
joint_model/dense_82/BiasAddBiasAdd%joint_model/dense_82/MatMul:product:03joint_model/dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
*joint_model/dense_83/MatMul/ReadVariableOpReadVariableOp3joint_model_dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0▓
joint_model/dense_83/MatMulMatMul%joint_model/dense_82/BiasAdd:output:02joint_model/dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+joint_model/dense_83/BiasAdd/ReadVariableOpReadVariableOp4joint_model_dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
joint_model/dense_83/BiasAddBiasAdd%joint_model/dense_83/MatMul:product:03joint_model/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ┤
5joint_model/prediction_output_0/MatMul/ReadVariableOpReadVariableOp>joint_model_prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╚
&joint_model/prediction_output_0/MatMulMatMul%joint_model/dense_83/BiasAdd:output:0=joint_model/prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
6joint_model/prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp?joint_model_prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╓
'joint_model/prediction_output_0/BiasAddBiasAdd0joint_model/prediction_output_0/MatMul:product:0>joint_model/prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
IdentityIdentity0joint_model/prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         °	
NoOpNoOpC^joint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOpE^joint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12^joint_model/batch_normalization_27/ReadVariableOp4^joint_model/batch_normalization_27/ReadVariableOp_1.^joint_model/conv2d_135/BiasAdd/ReadVariableOp-^joint_model/conv2d_135/Conv2D/ReadVariableOp.^joint_model/conv2d_136/BiasAdd/ReadVariableOp-^joint_model/conv2d_136/Conv2D/ReadVariableOp.^joint_model/conv2d_137/BiasAdd/ReadVariableOp-^joint_model/conv2d_137/Conv2D/ReadVariableOp.^joint_model/conv2d_138/BiasAdd/ReadVariableOp-^joint_model/conv2d_138/Conv2D/ReadVariableOp.^joint_model/conv2d_139/BiasAdd/ReadVariableOp-^joint_model/conv2d_139/Conv2D/ReadVariableOp.^joint_model/conv2d_140/BiasAdd/ReadVariableOp-^joint_model/conv2d_140/Conv2D/ReadVariableOp,^joint_model/dense_81/BiasAdd/ReadVariableOp+^joint_model/dense_81/MatMul/ReadVariableOp,^joint_model/dense_82/BiasAdd/ReadVariableOp+^joint_model/dense_82/MatMul/ReadVariableOp,^joint_model/dense_83/BiasAdd/ReadVariableOp+^joint_model/dense_83/MatMul/ReadVariableOp7^joint_model/prediction_output_0/BiasAdd/ReadVariableOp6^joint_model/prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2И
Bjoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOpBjoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2М
Djoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Djoint_model/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12f
1joint_model/batch_normalization_27/ReadVariableOp1joint_model/batch_normalization_27/ReadVariableOp2j
3joint_model/batch_normalization_27/ReadVariableOp_13joint_model/batch_normalization_27/ReadVariableOp_12^
-joint_model/conv2d_135/BiasAdd/ReadVariableOp-joint_model/conv2d_135/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_135/Conv2D/ReadVariableOp,joint_model/conv2d_135/Conv2D/ReadVariableOp2^
-joint_model/conv2d_136/BiasAdd/ReadVariableOp-joint_model/conv2d_136/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_136/Conv2D/ReadVariableOp,joint_model/conv2d_136/Conv2D/ReadVariableOp2^
-joint_model/conv2d_137/BiasAdd/ReadVariableOp-joint_model/conv2d_137/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_137/Conv2D/ReadVariableOp,joint_model/conv2d_137/Conv2D/ReadVariableOp2^
-joint_model/conv2d_138/BiasAdd/ReadVariableOp-joint_model/conv2d_138/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_138/Conv2D/ReadVariableOp,joint_model/conv2d_138/Conv2D/ReadVariableOp2^
-joint_model/conv2d_139/BiasAdd/ReadVariableOp-joint_model/conv2d_139/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_139/Conv2D/ReadVariableOp,joint_model/conv2d_139/Conv2D/ReadVariableOp2^
-joint_model/conv2d_140/BiasAdd/ReadVariableOp-joint_model/conv2d_140/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_140/Conv2D/ReadVariableOp,joint_model/conv2d_140/Conv2D/ReadVariableOp2Z
+joint_model/dense_81/BiasAdd/ReadVariableOp+joint_model/dense_81/BiasAdd/ReadVariableOp2X
*joint_model/dense_81/MatMul/ReadVariableOp*joint_model/dense_81/MatMul/ReadVariableOp2Z
+joint_model/dense_82/BiasAdd/ReadVariableOp+joint_model/dense_82/BiasAdd/ReadVariableOp2X
*joint_model/dense_82/MatMul/ReadVariableOp*joint_model/dense_82/MatMul/ReadVariableOp2Z
+joint_model/dense_83/BiasAdd/ReadVariableOp+joint_model/dense_83/BiasAdd/ReadVariableOp2X
*joint_model/dense_83/MatMul/ReadVariableOp*joint_model/dense_83/MatMul/ReadVariableOp2p
6joint_model/prediction_output_0/BiasAdd/ReadVariableOp6joint_model/prediction_output_0/BiasAdd/ReadVariableOp2n
5joint_model/prediction_output_0/MatMul/ReadVariableOp5joint_model/prediction_output_0/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         
"
_user_specified_name
input_46:YU
/
_output_shapes
:         
"
_user_specified_name
input_47
╝
u
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1348356

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         TW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         T"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         P:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         P
 
_user_specified_nameinputs
фO
И
H__inference_joint_model_layer_call_and_return_conditional_losses_1348719

inputs
inputs_1,
conv2d_138_1348654:		 
conv2d_138_1348656:,
conv2d_135_1348659: 
conv2d_135_1348661:,
batch_normalization_27_1348665:,
batch_normalization_27_1348667:,
batch_normalization_27_1348669:,
batch_normalization_27_1348671:,
conv2d_139_1348674: 
conv2d_139_1348676:,
conv2d_136_1348679:	 
conv2d_136_1348681:,
conv2d_140_1348684: 
conv2d_140_1348686:,
conv2d_137_1348689: 
conv2d_137_1348691:"
dense_81_1348698:T
dense_81_1348700:"
dense_82_1348703:
dense_82_1348705:"
dense_83_1348708:
dense_83_1348710:-
prediction_output_0_1348713:)
prediction_output_0_1348715:
identityИв.batch_normalization_27/StatefulPartitionedCallв"conv2d_135/StatefulPartitionedCallв"conv2d_136/StatefulPartitionedCallв"conv2d_137/StatefulPartitionedCallв"conv2d_138/StatefulPartitionedCallв"conv2d_139/StatefulPartitionedCallв"conv2d_140/StatefulPartitionedCallв dense_81/StatefulPartitionedCallв dense_82/StatefulPartitionedCallв dense_83/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallв,spatial_dropout2d_18/StatefulPartitionedCallИ
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_138_1348654conv2d_138_1348656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1348236Ж
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_135_1348659conv2d_135_1348661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1348252С
,spatial_dropout2d_18/StatefulPartitionedCallStatefulPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348137Э
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_27_1348665batch_normalization_27_1348667batch_normalization_27_1348669batch_normalization_27_1348671*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348193╡
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall5spatial_dropout2d_18/StatefulPartitionedCall:output:0conv2d_139_1348674conv2d_139_1348676*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1348278╖
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_136_1348679conv2d_136_1348681*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1348294л
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0conv2d_140_1348684conv2d_140_1348686*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1348310л
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0conv2d_137_1348689conv2d_137_1348691*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1348326 
'global_max_pooling2d_18/PartitionedCallPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1348214ъ
flatten_46/PartitionedCallPartitionedCall0global_max_pooling2d_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_1348339х
flatten_45/PartitionedCallPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_1348347Л
concatenate_18/PartitionedCallPartitionedCall#flatten_46/PartitionedCall:output:0#flatten_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1348356Ч
 dense_81/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0dense_81_1348698dense_81_1348700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_1348368Щ
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_1348703dense_82_1348705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1348384Щ
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1348708dense_83_1348710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1348400┼
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0prediction_output_0_1348713prediction_output_0_1348715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1348416Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ы
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall-^spatial_dropout2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2\
,spatial_dropout2d_18/StatefulPartitionedCall,spatial_dropout2d_18/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
╙	
Б
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1348416

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є
б
,__inference_conv2d_137_layer_call_fn_1349541

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1348326w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
к

А
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1348252

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╖
c
G__inference_flatten_46_layer_call_and_return_conditional_losses_1349562

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╬
Ю
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348162

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц	
╙
8__inference_batch_normalization_27_layer_call_fn_1349447

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348193Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
Ч
*__inference_dense_83_layer_call_fn_1349633

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1348400o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к

А
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1349421

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         


 
_user_specified_nameinputs
Щ
o
6__inference_spatial_dropout2d_18_layer_call_fn_1349355

inputs
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348137Т
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4                                    `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╚	
Ў
E__inference_dense_81_layer_call_and_return_conditional_losses_1348368

inputs0
matmul_readvariableop_resource:T-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
╖
c
G__inference_flatten_46_layer_call_and_return_conditional_losses_1348339

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к

А
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1348310

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
к

А
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1349502

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╚	
Ў
E__inference_dense_82_layer_call_and_return_conditional_losses_1349624

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╢
H
,__inference_flatten_45_layer_call_fn_1349567

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_1348347`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ё
o
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348109

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4                                    ~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4                                    "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к

А
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1348236

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         

w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▌
в
5__inference_prediction_output_0_layer_call_fn_1349652

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1348416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к

А
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1348294

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ьO
К
H__inference_joint_model_layer_call_and_return_conditional_losses_1348962
input_46
input_47,
conv2d_138_1348897:		 
conv2d_138_1348899:,
conv2d_135_1348902: 
conv2d_135_1348904:,
batch_normalization_27_1348908:,
batch_normalization_27_1348910:,
batch_normalization_27_1348912:,
batch_normalization_27_1348914:,
conv2d_139_1348917: 
conv2d_139_1348919:,
conv2d_136_1348922:	 
conv2d_136_1348924:,
conv2d_140_1348927: 
conv2d_140_1348929:,
conv2d_137_1348932: 
conv2d_137_1348934:"
dense_81_1348941:T
dense_81_1348943:"
dense_82_1348946:
dense_82_1348948:"
dense_83_1348951:
dense_83_1348953:-
prediction_output_0_1348956:)
prediction_output_0_1348958:
identityИв.batch_normalization_27/StatefulPartitionedCallв"conv2d_135/StatefulPartitionedCallв"conv2d_136/StatefulPartitionedCallв"conv2d_137/StatefulPartitionedCallв"conv2d_138/StatefulPartitionedCallв"conv2d_139/StatefulPartitionedCallв"conv2d_140/StatefulPartitionedCallв dense_81/StatefulPartitionedCallв dense_82/StatefulPartitionedCallв dense_83/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallв,spatial_dropout2d_18/StatefulPartitionedCallИ
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCallinput_47conv2d_138_1348897conv2d_138_1348899*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1348236И
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCallinput_46conv2d_135_1348902conv2d_135_1348904*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1348252С
,spatial_dropout2d_18/StatefulPartitionedCallStatefulPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348137Э
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_27_1348908batch_normalization_27_1348910batch_normalization_27_1348912batch_normalization_27_1348914*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348193╡
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall5spatial_dropout2d_18/StatefulPartitionedCall:output:0conv2d_139_1348917conv2d_139_1348919*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1348278╖
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_136_1348922conv2d_136_1348924*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1348294л
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0conv2d_140_1348927conv2d_140_1348929*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1348310л
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0conv2d_137_1348932conv2d_137_1348934*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1348326 
'global_max_pooling2d_18/PartitionedCallPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1348214ъ
flatten_46/PartitionedCallPartitionedCall0global_max_pooling2d_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_1348339х
flatten_45/PartitionedCallPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_1348347Л
concatenate_18/PartitionedCallPartitionedCall#flatten_46/PartitionedCall:output:0#flatten_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1348356Ч
 dense_81/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0dense_81_1348941dense_81_1348943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_1348368Щ
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_1348946dense_82_1348948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1348384Щ
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1348951dense_83_1348953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1348400┼
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0prediction_output_0_1348956prediction_output_0_1348958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1348416Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ы
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall-^spatial_dropout2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2\
,spatial_dropout2d_18/StatefulPartitionedCall,spatial_dropout2d_18/StatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_46:YU
/
_output_shapes
:         
"
_user_specified_name
input_47
╚	
Ў
E__inference_dense_81_layer_call_and_return_conditional_losses_1349605

inputs0
matmul_readvariableop_resource:T-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
к

А
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1349551

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╬
Ю
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349465

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш	
╙
8__inference_batch_normalization_27_layer_call_fn_1349434

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348162Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
░
p
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1349532

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:                  ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к

А
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1349345

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         

w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
░
p
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1348214

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:                  ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
А
p
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349383

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?З
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4                                    `
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╓
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:м
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"                  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╖
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"                  А
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"                  М
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4                                    |
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ЪN
█
H__inference_joint_model_layer_call_and_return_conditional_losses_1348893
input_46
input_47,
conv2d_138_1348828:		 
conv2d_138_1348830:,
conv2d_135_1348833: 
conv2d_135_1348835:,
batch_normalization_27_1348839:,
batch_normalization_27_1348841:,
batch_normalization_27_1348843:,
batch_normalization_27_1348845:,
conv2d_139_1348848: 
conv2d_139_1348850:,
conv2d_136_1348853:	 
conv2d_136_1348855:,
conv2d_140_1348858: 
conv2d_140_1348860:,
conv2d_137_1348863: 
conv2d_137_1348865:"
dense_81_1348872:T
dense_81_1348874:"
dense_82_1348877:
dense_82_1348879:"
dense_83_1348882:
dense_83_1348884:-
prediction_output_0_1348887:)
prediction_output_0_1348889:
identityИв.batch_normalization_27/StatefulPartitionedCallв"conv2d_135/StatefulPartitionedCallв"conv2d_136/StatefulPartitionedCallв"conv2d_137/StatefulPartitionedCallв"conv2d_138/StatefulPartitionedCallв"conv2d_139/StatefulPartitionedCallв"conv2d_140/StatefulPartitionedCallв dense_81/StatefulPartitionedCallв dense_82/StatefulPartitionedCallв dense_83/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallИ
"conv2d_138/StatefulPartitionedCallStatefulPartitionedCallinput_47conv2d_138_1348828conv2d_138_1348830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1348236И
"conv2d_135/StatefulPartitionedCallStatefulPartitionedCallinput_46conv2d_135_1348833conv2d_135_1348835*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1348252Б
$spatial_dropout2d_18/PartitionedCallPartitionedCall+conv2d_138/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348109Я
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall+conv2d_135/StatefulPartitionedCall:output:0batch_normalization_27_1348839batch_normalization_27_1348841batch_normalization_27_1348843batch_normalization_27_1348845*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1348162н
"conv2d_139/StatefulPartitionedCallStatefulPartitionedCall-spatial_dropout2d_18/PartitionedCall:output:0conv2d_139_1348848conv2d_139_1348850*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1348278╖
"conv2d_136/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_136_1348853conv2d_136_1348855*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1348294л
"conv2d_140/StatefulPartitionedCallStatefulPartitionedCall+conv2d_139/StatefulPartitionedCall:output:0conv2d_140_1348858conv2d_140_1348860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1348310л
"conv2d_137/StatefulPartitionedCallStatefulPartitionedCall+conv2d_136/StatefulPartitionedCall:output:0conv2d_137_1348863conv2d_137_1348865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1348326 
'global_max_pooling2d_18/PartitionedCallPartitionedCall+conv2d_140/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1348214ъ
flatten_46/PartitionedCallPartitionedCall0global_max_pooling2d_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_1348339х
flatten_45/PartitionedCallPartitionedCall+conv2d_137/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_1348347Л
concatenate_18/PartitionedCallPartitionedCall#flatten_46/PartitionedCall:output:0#flatten_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1348356Ч
 dense_81/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0dense_81_1348872dense_81_1348874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_1348368Щ
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_1348877dense_82_1348879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1348384Щ
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1348882dense_83_1348884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1348400┼
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0prediction_output_0_1348887prediction_output_0_1348889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1348416Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ь
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall#^conv2d_135/StatefulPartitionedCall#^conv2d_136/StatefulPartitionedCall#^conv2d_137/StatefulPartitionedCall#^conv2d_138/StatefulPartitionedCall#^conv2d_139/StatefulPartitionedCall#^conv2d_140/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2H
"conv2d_135/StatefulPartitionedCall"conv2d_135/StatefulPartitionedCall2H
"conv2d_136/StatefulPartitionedCall"conv2d_136/StatefulPartitionedCall2H
"conv2d_137/StatefulPartitionedCall"conv2d_137/StatefulPartitionedCall2H
"conv2d_138/StatefulPartitionedCall"conv2d_138/StatefulPartitionedCall2H
"conv2d_139/StatefulPartitionedCall"conv2d_139/StatefulPartitionedCall2H
"conv2d_140/StatefulPartitionedCall"conv2d_140/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_46:YU
/
_output_shapes
:         
"
_user_specified_name
input_47
╨
▒
-__inference_joint_model_layer_call_fn_1349078
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:T

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_1348423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
╚	
Ў
E__inference_dense_82_layer_call_and_return_conditional_losses_1348384

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є
б
,__inference_conv2d_135_layer_call_fn_1349392

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1348252w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╟
Ч
*__inference_dense_82_layer_call_fn_1349614

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1348384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
їФ
╨ 
 __inference__traced_save_1349911
file_prefix0
,savev2_conv2d_138_kernel_read_readvariableop.
*savev2_conv2d_138_bias_read_readvariableop0
,savev2_conv2d_135_kernel_read_readvariableop.
*savev2_conv2d_135_bias_read_readvariableop0
,savev2_conv2d_139_kernel_read_readvariableop.
*savev2_conv2d_139_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop0
,savev2_conv2d_140_kernel_read_readvariableop.
*savev2_conv2d_140_bias_read_readvariableop0
,savev2_conv2d_136_kernel_read_readvariableop.
*savev2_conv2d_136_bias_read_readvariableop0
,savev2_conv2d_137_kernel_read_readvariableop.
*savev2_conv2d_137_bias_read_readvariableop.
*savev2_dense_81_kernel_read_readvariableop,
(savev2_dense_81_bias_read_readvariableop.
*savev2_dense_82_kernel_read_readvariableop,
(savev2_dense_82_bias_read_readvariableop.
*savev2_dense_83_kernel_read_readvariableop,
(savev2_dense_83_bias_read_readvariableop9
5savev2_prediction_output_0_kernel_read_readvariableop7
3savev2_prediction_output_0_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_138_kernel_m_read_readvariableop5
1savev2_adam_conv2d_138_bias_m_read_readvariableop7
3savev2_adam_conv2d_135_kernel_m_read_readvariableop5
1savev2_adam_conv2d_135_bias_m_read_readvariableop7
3savev2_adam_conv2d_139_kernel_m_read_readvariableop5
1savev2_adam_conv2d_139_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_m_read_readvariableop7
3savev2_adam_conv2d_140_kernel_m_read_readvariableop5
1savev2_adam_conv2d_140_bias_m_read_readvariableop7
3savev2_adam_conv2d_136_kernel_m_read_readvariableop5
1savev2_adam_conv2d_136_bias_m_read_readvariableop7
3savev2_adam_conv2d_137_kernel_m_read_readvariableop5
1savev2_adam_conv2d_137_bias_m_read_readvariableop5
1savev2_adam_dense_81_kernel_m_read_readvariableop3
/savev2_adam_dense_81_bias_m_read_readvariableop5
1savev2_adam_dense_82_kernel_m_read_readvariableop3
/savev2_adam_dense_82_bias_m_read_readvariableop5
1savev2_adam_dense_83_kernel_m_read_readvariableop3
/savev2_adam_dense_83_bias_m_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_m_read_readvariableop>
:savev2_adam_prediction_output_0_bias_m_read_readvariableop7
3savev2_adam_conv2d_138_kernel_v_read_readvariableop5
1savev2_adam_conv2d_138_bias_v_read_readvariableop7
3savev2_adam_conv2d_135_kernel_v_read_readvariableop5
1savev2_adam_conv2d_135_bias_v_read_readvariableop7
3savev2_adam_conv2d_139_kernel_v_read_readvariableop5
1savev2_adam_conv2d_139_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_v_read_readvariableop7
3savev2_adam_conv2d_140_kernel_v_read_readvariableop5
1savev2_adam_conv2d_140_bias_v_read_readvariableop7
3savev2_adam_conv2d_136_kernel_v_read_readvariableop5
1savev2_adam_conv2d_136_bias_v_read_readvariableop7
3savev2_adam_conv2d_137_kernel_v_read_readvariableop5
1savev2_adam_conv2d_137_bias_v_read_readvariableop5
1savev2_adam_dense_81_kernel_v_read_readvariableop3
/savev2_adam_dense_81_bias_v_read_readvariableop5
1savev2_adam_dense_82_kernel_v_read_readvariableop3
/savev2_adam_dense_82_bias_v_read_readvariableop5
1savev2_adam_dense_83_kernel_v_read_readvariableop3
/savev2_adam_dense_83_bias_v_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_v_read_readvariableop>
:savev2_adam_prediction_output_0_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┌*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*Г*
value∙)BЎ)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHИ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*н
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ▓
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_138_kernel_read_readvariableop*savev2_conv2d_138_bias_read_readvariableop,savev2_conv2d_135_kernel_read_readvariableop*savev2_conv2d_135_bias_read_readvariableop,savev2_conv2d_139_kernel_read_readvariableop*savev2_conv2d_139_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop,savev2_conv2d_140_kernel_read_readvariableop*savev2_conv2d_140_bias_read_readvariableop,savev2_conv2d_136_kernel_read_readvariableop*savev2_conv2d_136_bias_read_readvariableop,savev2_conv2d_137_kernel_read_readvariableop*savev2_conv2d_137_bias_read_readvariableop*savev2_dense_81_kernel_read_readvariableop(savev2_dense_81_bias_read_readvariableop*savev2_dense_82_kernel_read_readvariableop(savev2_dense_82_bias_read_readvariableop*savev2_dense_83_kernel_read_readvariableop(savev2_dense_83_bias_read_readvariableop5savev2_prediction_output_0_kernel_read_readvariableop3savev2_prediction_output_0_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_138_kernel_m_read_readvariableop1savev2_adam_conv2d_138_bias_m_read_readvariableop3savev2_adam_conv2d_135_kernel_m_read_readvariableop1savev2_adam_conv2d_135_bias_m_read_readvariableop3savev2_adam_conv2d_139_kernel_m_read_readvariableop1savev2_adam_conv2d_139_bias_m_read_readvariableop>savev2_adam_batch_normalization_27_gamma_m_read_readvariableop=savev2_adam_batch_normalization_27_beta_m_read_readvariableop3savev2_adam_conv2d_140_kernel_m_read_readvariableop1savev2_adam_conv2d_140_bias_m_read_readvariableop3savev2_adam_conv2d_136_kernel_m_read_readvariableop1savev2_adam_conv2d_136_bias_m_read_readvariableop3savev2_adam_conv2d_137_kernel_m_read_readvariableop1savev2_adam_conv2d_137_bias_m_read_readvariableop1savev2_adam_dense_81_kernel_m_read_readvariableop/savev2_adam_dense_81_bias_m_read_readvariableop1savev2_adam_dense_82_kernel_m_read_readvariableop/savev2_adam_dense_82_bias_m_read_readvariableop1savev2_adam_dense_83_kernel_m_read_readvariableop/savev2_adam_dense_83_bias_m_read_readvariableop<savev2_adam_prediction_output_0_kernel_m_read_readvariableop:savev2_adam_prediction_output_0_bias_m_read_readvariableop3savev2_adam_conv2d_138_kernel_v_read_readvariableop1savev2_adam_conv2d_138_bias_v_read_readvariableop3savev2_adam_conv2d_135_kernel_v_read_readvariableop1savev2_adam_conv2d_135_bias_v_read_readvariableop3savev2_adam_conv2d_139_kernel_v_read_readvariableop1savev2_adam_conv2d_139_bias_v_read_readvariableop>savev2_adam_batch_normalization_27_gamma_v_read_readvariableop=savev2_adam_batch_normalization_27_beta_v_read_readvariableop3savev2_adam_conv2d_140_kernel_v_read_readvariableop1savev2_adam_conv2d_140_bias_v_read_readvariableop3savev2_adam_conv2d_136_kernel_v_read_readvariableop1savev2_adam_conv2d_136_bias_v_read_readvariableop3savev2_adam_conv2d_137_kernel_v_read_readvariableop1savev2_adam_conv2d_137_bias_v_read_readvariableop1savev2_adam_dense_81_kernel_v_read_readvariableop/savev2_adam_dense_81_bias_v_read_readvariableop1savev2_adam_dense_82_kernel_v_read_readvariableop/savev2_adam_dense_82_bias_v_read_readvariableop1savev2_adam_dense_83_kernel_v_read_readvariableop/savev2_adam_dense_83_bias_v_read_readvariableop<savev2_adam_prediction_output_0_kernel_v_read_readvariableop:savev2_adam_prediction_output_0_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*╟
_input_shapes╡
▓: :		::::::::::::	::::T:::::::: : : : : : : :		::::::::::	::::T::::::::		::::::::::	::::T:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:T: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
:		: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:	: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::$. 

_output_shapes

:T: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::,6(
&
_output_shapes
:		: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:	: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
::$D 

_output_shapes

:T: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::$J 

_output_shapes

:: K

_output_shapes
::L

_output_shapes
: 
в
й
%__inference_signature_wrapper_1349024
input_46
input_47!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:T

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_46input_47unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_1348100o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:         :         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
input_46:YU
/
_output_shapes
:         
"
_user_specified_name
input_47
к

А
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1349521

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╟
c
G__inference_flatten_45_layer_call_and_return_conditional_losses_1349573

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    P   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         PX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Но
╘0
#__inference__traced_restore_1350146
file_prefix<
"assignvariableop_conv2d_138_kernel:		0
"assignvariableop_1_conv2d_138_bias:>
$assignvariableop_2_conv2d_135_kernel:0
"assignvariableop_3_conv2d_135_bias:>
$assignvariableop_4_conv2d_139_kernel:0
"assignvariableop_5_conv2d_139_bias:=
/assignvariableop_6_batch_normalization_27_gamma:<
.assignvariableop_7_batch_normalization_27_beta:C
5assignvariableop_8_batch_normalization_27_moving_mean:G
9assignvariableop_9_batch_normalization_27_moving_variance:?
%assignvariableop_10_conv2d_140_kernel:1
#assignvariableop_11_conv2d_140_bias:?
%assignvariableop_12_conv2d_136_kernel:	1
#assignvariableop_13_conv2d_136_bias:?
%assignvariableop_14_conv2d_137_kernel:1
#assignvariableop_15_conv2d_137_bias:5
#assignvariableop_16_dense_81_kernel:T/
!assignvariableop_17_dense_81_bias:5
#assignvariableop_18_dense_82_kernel:/
!assignvariableop_19_dense_82_bias:5
#assignvariableop_20_dense_83_kernel:/
!assignvariableop_21_dense_83_bias:@
.assignvariableop_22_prediction_output_0_kernel::
,assignvariableop_23_prediction_output_0_bias:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: #
assignvariableop_29_total: #
assignvariableop_30_count: F
,assignvariableop_31_adam_conv2d_138_kernel_m:		8
*assignvariableop_32_adam_conv2d_138_bias_m:F
,assignvariableop_33_adam_conv2d_135_kernel_m:8
*assignvariableop_34_adam_conv2d_135_bias_m:F
,assignvariableop_35_adam_conv2d_139_kernel_m:8
*assignvariableop_36_adam_conv2d_139_bias_m:E
7assignvariableop_37_adam_batch_normalization_27_gamma_m:D
6assignvariableop_38_adam_batch_normalization_27_beta_m:F
,assignvariableop_39_adam_conv2d_140_kernel_m:8
*assignvariableop_40_adam_conv2d_140_bias_m:F
,assignvariableop_41_adam_conv2d_136_kernel_m:	8
*assignvariableop_42_adam_conv2d_136_bias_m:F
,assignvariableop_43_adam_conv2d_137_kernel_m:8
*assignvariableop_44_adam_conv2d_137_bias_m:<
*assignvariableop_45_adam_dense_81_kernel_m:T6
(assignvariableop_46_adam_dense_81_bias_m:<
*assignvariableop_47_adam_dense_82_kernel_m:6
(assignvariableop_48_adam_dense_82_bias_m:<
*assignvariableop_49_adam_dense_83_kernel_m:6
(assignvariableop_50_adam_dense_83_bias_m:G
5assignvariableop_51_adam_prediction_output_0_kernel_m:A
3assignvariableop_52_adam_prediction_output_0_bias_m:F
,assignvariableop_53_adam_conv2d_138_kernel_v:		8
*assignvariableop_54_adam_conv2d_138_bias_v:F
,assignvariableop_55_adam_conv2d_135_kernel_v:8
*assignvariableop_56_adam_conv2d_135_bias_v:F
,assignvariableop_57_adam_conv2d_139_kernel_v:8
*assignvariableop_58_adam_conv2d_139_bias_v:E
7assignvariableop_59_adam_batch_normalization_27_gamma_v:D
6assignvariableop_60_adam_batch_normalization_27_beta_v:F
,assignvariableop_61_adam_conv2d_140_kernel_v:8
*assignvariableop_62_adam_conv2d_140_bias_v:F
,assignvariableop_63_adam_conv2d_136_kernel_v:	8
*assignvariableop_64_adam_conv2d_136_bias_v:F
,assignvariableop_65_adam_conv2d_137_kernel_v:8
*assignvariableop_66_adam_conv2d_137_bias_v:<
*assignvariableop_67_adam_dense_81_kernel_v:T6
(assignvariableop_68_adam_dense_81_bias_v:<
*assignvariableop_69_adam_dense_82_kernel_v:6
(assignvariableop_70_adam_dense_82_bias_v:<
*assignvariableop_71_adam_dense_83_kernel_v:6
(assignvariableop_72_adam_dense_83_bias_v:G
5assignvariableop_73_adam_prediction_output_0_kernel_v:A
3assignvariableop_74_adam_prediction_output_0_bias_v:
identity_76ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_8вAssignVariableOp_9▌*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*Г*
value∙)BЎ)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЛ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*н
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Э
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╞
_output_shapes│
░::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_138_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_138_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_135_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_135_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_139_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_139_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_27_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_27_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_27_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_27_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_140_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_140_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_136_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_136_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_137_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_137_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_81_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_81_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_82_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_82_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_83_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_83_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_22AssignVariableOp.assignvariableop_22_prediction_output_0_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp,assignvariableop_23_prediction_output_0_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_138_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_138_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_135_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_135_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_139_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_139_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_27_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_27_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_140_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_140_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv2d_136_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_136_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_137_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_137_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_81_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_81_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_82_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_82_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_83_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_83_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_prediction_output_0_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_prediction_output_0_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_138_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_138_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_135_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_135_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_139_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_139_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_27_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_27_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_140_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_140_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_136_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_136_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_137_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_137_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_81_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_81_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_82_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_82_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_83_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_83_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_prediction_output_0_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adam_prediction_output_0_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ┴
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: о
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*н
_input_shapesЫ
Ш: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╟
c
G__inference_flatten_45_layer_call_and_return_conditional_losses_1348347

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    P   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         PX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╟
R
6__inference_spatial_dropout2d_18_layer_call_fn_1349350

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1348109Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
є
б
,__inference_conv2d_140_layer_call_fn_1349492

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1348310w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─
w
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1349586
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         TW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         T"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         P:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         P
"
_user_specified_name
inputs/1
к

А
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1349402

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ё
o
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349360

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4                                    ~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4                                    "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*З
serving_defaultє
E
input_469
serving_default_input_46:0         
E
input_479
serving_default_input_47:0         G
prediction_output_00
StatefulPartitionedCall:0         tensorflow/serving/predict:╥И
Ї
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op"
_tf_keras_layer
"
_tf_keras_input_layer
╝
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator"
_tf_keras_layer
▌
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
▌
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
ъ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance"
_tf_keras_layer
▌
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op"
_tf_keras_layer
▌
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op"
_tf_keras_layer
е
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op"
_tf_keras_layer
е
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
е
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
е
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
┐
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias"
_tf_keras_layer
├
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
Кkernel
	Лbias"
_tf_keras_layer
├
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkernel
	Уbias"
_tf_keras_layer
├
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъkernel
	Ыbias"
_tf_keras_layer
▐
"0
#1
22
33
;4
<5
E6
F7
G8
H9
O10
P11
X12
Y13
g14
h15
В16
Г17
К18
Л19
Т20
У21
Ъ22
Ы23"
trackable_list_wrapper
╬
"0
#1
22
33
;4
<5
E6
F7
O8
P9
X10
Y11
g12
h13
В14
Г15
К16
Л17
Т18
У19
Ъ20
Ы21"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
бtrace_0
вtrace_1
гtrace_2
дtrace_32■
-__inference_joint_model_layer_call_fn_1348474
-__inference_joint_model_layer_call_fn_1349078
-__inference_joint_model_layer_call_fn_1349132
-__inference_joint_model_layer_call_fn_1348824┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0zвtrace_1zгtrace_2zдtrace_3
▌
еtrace_0
жtrace_1
зtrace_2
иtrace_32ъ
H__inference_joint_model_layer_call_and_return_conditional_losses_1349220
H__inference_joint_model_layer_call_and_return_conditional_losses_1349326
H__inference_joint_model_layer_call_and_return_conditional_losses_1348893
H__inference_joint_model_layer_call_and_return_conditional_losses_1348962┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0zжtrace_1zзtrace_2zиtrace_3
╪B╒
"__inference__wrapped_model_1348100input_46input_47"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
а
	йiter
кbeta_1
лbeta_2

мdecay
нlearning_rate"mи#mй2mк3mл;mм<mнEmоFmпOm░Pm▒Xm▓Ym│gm┤hm╡	Вm╢	Гm╖	Кm╕	Лm╣	Тm║	Уm╗	Ъm╝	Ыm╜"v╛#v┐2v└3v┴;v┬<v├Ev─Fv┼Ov╞Pv╟Xv╚Yv╔gv╩hv╦	Вv╠	Гv═	Кv╬	Лv╧	Тv╨	Уv╤	Ъv╥	Ыv╙"
	optimizer
-
оserving_default"
signature_map
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Є
┤trace_02╙
,__inference_conv2d_138_layer_call_fn_1349335в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0
Н
╡trace_02ю
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1349345в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╡trace_0
+:)		2conv2d_138/kernel
:2conv2d_138/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
с
╗trace_0
╝trace_12ж
6__inference_spatial_dropout2d_18_layer_call_fn_1349350
6__inference_spatial_dropout2d_18_layer_call_fn_1349355│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0z╝trace_1
Ч
╜trace_0
╛trace_12▄
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349360
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349383│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0z╛trace_1
"
_generic_user_object
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Є
─trace_02╙
,__inference_conv2d_135_layer_call_fn_1349392в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
Н
┼trace_02ю
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1349402в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0
+:)2conv2d_135/kernel
:2conv2d_135/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Є
╦trace_02╙
,__inference_conv2d_139_layer_call_fn_1349411в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
Н
╠trace_02ю
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1349421в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0
+:)2conv2d_139/kernel
:2conv2d_139/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
х
╥trace_0
╙trace_12к
8__inference_batch_normalization_27_layer_call_fn_1349434
8__inference_batch_normalization_27_layer_call_fn_1349447│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0z╙trace_1
Ы
╘trace_0
╒trace_12р
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349465
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349483│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╘trace_0z╒trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_27/gamma
):'2batch_normalization_27/beta
2:0 (2"batch_normalization_27/moving_mean
6:4 (2&batch_normalization_27/moving_variance
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
Є
█trace_02╙
,__inference_conv2d_140_layer_call_fn_1349492в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z█trace_0
Н
▄trace_02ю
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1349502в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▄trace_0
+:)2conv2d_140/kernel
:2conv2d_140/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Є
тtrace_02╙
,__inference_conv2d_136_layer_call_fn_1349511в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0
Н
уtrace_02ю
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1349521в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zуtrace_0
+:)	2conv2d_136/kernel
:2conv2d_136/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 
щtrace_02р
9__inference_global_max_pooling2d_18_layer_call_fn_1349526в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
Ъ
ъtrace_02√
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1349532в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zъtrace_0
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Є
Ёtrace_02╙
,__inference_conv2d_137_layer_call_fn_1349541в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
Н
ёtrace_02ю
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1349551в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zёtrace_0
+:)2conv2d_137/kernel
:2conv2d_137/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Є
ўtrace_02╙
,__inference_flatten_46_layer_call_fn_1349556в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
Н
°trace_02ю
G__inference_flatten_46_layer_call_and_return_conditional_losses_1349562в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
Є
■trace_02╙
,__inference_flatten_45_layer_call_fn_1349567в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
Н
 trace_02ю
G__inference_flatten_45_layer_call_and_return_conditional_losses_1349573в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ў
Еtrace_02╫
0__inference_concatenate_18_layer_call_fn_1349579в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
С
Жtrace_02Є
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1349586в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
Ё
Мtrace_02╤
*__inference_dense_81_layer_call_fn_1349595в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
Л
Нtrace_02ь
E__inference_dense_81_layer_call_and_return_conditional_losses_1349605в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
!:T2dense_81/kernel
:2dense_81/bias
0
К0
Л1"
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
Ё
Уtrace_02╤
*__inference_dense_82_layer_call_fn_1349614в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
Л
Фtrace_02ь
E__inference_dense_82_layer_call_and_return_conditional_losses_1349624в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
!:2dense_82/kernel
:2dense_82/bias
0
Т0
У1"
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
Ё
Ъtrace_02╤
*__inference_dense_83_layer_call_fn_1349633в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
Л
Ыtrace_02ь
E__inference_dense_83_layer_call_and_return_conditional_losses_1349643в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
!:2dense_83/kernel
:2dense_83/bias
0
Ъ0
Ы1"
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
√
бtrace_02▄
5__inference_prediction_output_0_layer_call_fn_1349652в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
Ц
вtrace_02ў
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1349662в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
,:*2prediction_output_0/kernel
&:$2prediction_output_0/bias
.
G0
H1"
trackable_list_wrapper
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
(
г0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
КBЗ
-__inference_joint_model_layer_call_fn_1348474input_46input_47"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
-__inference_joint_model_layer_call_fn_1349078inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
-__inference_joint_model_layer_call_fn_1349132inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
-__inference_joint_model_layer_call_fn_1348824input_46input_47"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
H__inference_joint_model_layer_call_and_return_conditional_losses_1349220inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
H__inference_joint_model_layer_call_and_return_conditional_losses_1349326inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
H__inference_joint_model_layer_call_and_return_conditional_losses_1348893input_46input_47"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
H__inference_joint_model_layer_call_and_return_conditional_losses_1348962input_46input_47"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╒B╥
%__inference_signature_wrapper_1349024input_46input_47"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_conv2d_138_layer_call_fn_1349335inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1349345inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
√B°
6__inference_spatial_dropout2d_18_layer_call_fn_1349350inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
6__inference_spatial_dropout2d_18_layer_call_fn_1349355inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349360inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349383inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_conv2d_135_layer_call_fn_1349392inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1349402inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_conv2d_139_layer_call_fn_1349411inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1349421inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¤B·
8__inference_batch_normalization_27_layer_call_fn_1349434inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
8__inference_batch_normalization_27_layer_call_fn_1349447inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349465inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349483inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_conv2d_140_layer_call_fn_1349492inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1349502inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_conv2d_136_layer_call_fn_1349511inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1349521inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
9__inference_global_max_pooling2d_18_layer_call_fn_1349526inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1349532inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_conv2d_137_layer_call_fn_1349541inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1349551inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_flatten_46_layer_call_fn_1349556inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_flatten_46_layer_call_and_return_conditional_losses_1349562inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_flatten_45_layer_call_fn_1349567inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_flatten_45_layer_call_and_return_conditional_losses_1349573inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBэ
0__inference_concatenate_18_layer_call_fn_1349579inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1349586inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_dense_81_layer_call_fn_1349595inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_81_layer_call_and_return_conditional_losses_1349605inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_dense_82_layer_call_fn_1349614inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_82_layer_call_and_return_conditional_losses_1349624inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_dense_83_layer_call_fn_1349633inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_83_layer_call_and_return_conditional_losses_1349643inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_prediction_output_0_layer_call_fn_1349652inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1349662inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
д	variables
е	keras_api

жtotal

зcount"
_tf_keras_metric
0
ж0
з1"
trackable_list_wrapper
.
д	variables"
_generic_user_object
:  (2total
:  (2count
0:.		2Adam/conv2d_138/kernel/m
": 2Adam/conv2d_138/bias/m
0:.2Adam/conv2d_135/kernel/m
": 2Adam/conv2d_135/bias/m
0:.2Adam/conv2d_139/kernel/m
": 2Adam/conv2d_139/bias/m
/:-2#Adam/batch_normalization_27/gamma/m
.:,2"Adam/batch_normalization_27/beta/m
0:.2Adam/conv2d_140/kernel/m
": 2Adam/conv2d_140/bias/m
0:.	2Adam/conv2d_136/kernel/m
": 2Adam/conv2d_136/bias/m
0:.2Adam/conv2d_137/kernel/m
": 2Adam/conv2d_137/bias/m
&:$T2Adam/dense_81/kernel/m
 :2Adam/dense_81/bias/m
&:$2Adam/dense_82/kernel/m
 :2Adam/dense_82/bias/m
&:$2Adam/dense_83/kernel/m
 :2Adam/dense_83/bias/m
1:/2!Adam/prediction_output_0/kernel/m
+:)2Adam/prediction_output_0/bias/m
0:.		2Adam/conv2d_138/kernel/v
": 2Adam/conv2d_138/bias/v
0:.2Adam/conv2d_135/kernel/v
": 2Adam/conv2d_135/bias/v
0:.2Adam/conv2d_139/kernel/v
": 2Adam/conv2d_139/bias/v
/:-2#Adam/batch_normalization_27/gamma/v
.:,2"Adam/batch_normalization_27/beta/v
0:.2Adam/conv2d_140/kernel/v
": 2Adam/conv2d_140/bias/v
0:.	2Adam/conv2d_136/kernel/v
": 2Adam/conv2d_136/bias/v
0:.2Adam/conv2d_137/kernel/v
": 2Adam/conv2d_137/bias/v
&:$T2Adam/dense_81/kernel/v
 :2Adam/dense_81/bias/v
&:$2Adam/dense_82/kernel/v
 :2Adam/dense_82/bias/v
&:$2Adam/dense_83/kernel/v
 :2Adam/dense_83/bias/v
1:/2!Adam/prediction_output_0/kernel/v
+:)2Adam/prediction_output_0/bias/vА
"__inference__wrapped_model_1348100┘ "#23EFGH;<XYOPghВГКЛТУЪЫjвg
`в]
[ЪX
*К'
input_46         
*К'
input_47         
к "IкF
D
prediction_output_0-К*
prediction_output_0         ю
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349465ЦEFGHMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ю
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1349483ЦEFGHMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╞
8__inference_batch_normalization_27_layer_call_fn_1349434ЙEFGHMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ╞
8__inference_batch_normalization_27_layer_call_fn_1349447ЙEFGHMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╙
K__inference_concatenate_18_layer_call_and_return_conditional_losses_1349586ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         P
к "%в"
К
0         T
Ъ к
0__inference_concatenate_18_layer_call_fn_1349579vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         P
к "К         T╖
G__inference_conv2d_135_layer_call_and_return_conditional_losses_1349402l237в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ П
,__inference_conv2d_135_layer_call_fn_1349392_237в4
-в*
(К%
inputs         
к " К         ╖
G__inference_conv2d_136_layer_call_and_return_conditional_losses_1349521lXY7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ П
,__inference_conv2d_136_layer_call_fn_1349511_XY7в4
-в*
(К%
inputs         
к " К         ╖
G__inference_conv2d_137_layer_call_and_return_conditional_losses_1349551lgh7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ П
,__inference_conv2d_137_layer_call_fn_1349541_gh7в4
-в*
(К%
inputs         
к " К         ╖
G__inference_conv2d_138_layer_call_and_return_conditional_losses_1349345l"#7в4
-в*
(К%
inputs         
к "-в*
#К 
0         


Ъ П
,__inference_conv2d_138_layer_call_fn_1349335_"#7в4
-в*
(К%
inputs         
к " К         

╖
G__inference_conv2d_139_layer_call_and_return_conditional_losses_1349421l;<7в4
-в*
(К%
inputs         


к "-в*
#К 
0         
Ъ П
,__inference_conv2d_139_layer_call_fn_1349411_;<7в4
-в*
(К%
inputs         


к " К         ╖
G__inference_conv2d_140_layer_call_and_return_conditional_losses_1349502lOP7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ П
,__inference_conv2d_140_layer_call_fn_1349492_OP7в4
-в*
(К%
inputs         
к " К         з
E__inference_dense_81_layer_call_and_return_conditional_losses_1349605^ВГ/в,
%в"
 К
inputs         T
к "%в"
К
0         
Ъ 
*__inference_dense_81_layer_call_fn_1349595QВГ/в,
%в"
 К
inputs         T
к "К         з
E__inference_dense_82_layer_call_and_return_conditional_losses_1349624^КЛ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ 
*__inference_dense_82_layer_call_fn_1349614QКЛ/в,
%в"
 К
inputs         
к "К         з
E__inference_dense_83_layer_call_and_return_conditional_losses_1349643^ТУ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ 
*__inference_dense_83_layer_call_fn_1349633QТУ/в,
%в"
 К
inputs         
к "К         л
G__inference_flatten_45_layer_call_and_return_conditional_losses_1349573`7в4
-в*
(К%
inputs         
к "%в"
К
0         P
Ъ Г
,__inference_flatten_45_layer_call_fn_1349567S7в4
-в*
(К%
inputs         
к "К         Pг
G__inference_flatten_46_layer_call_and_return_conditional_losses_1349562X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ {
,__inference_flatten_46_layer_call_fn_1349556K/в,
%в"
 К
inputs         
к "К         ▌
T__inference_global_max_pooling2d_18_layer_call_and_return_conditional_losses_1349532ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ┤
9__inference_global_max_pooling2d_18_layer_call_fn_1349526wRвO
HвE
CК@
inputs4                                    
к "!К                  К
H__inference_joint_model_layer_call_and_return_conditional_losses_1348893╜ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
input_46         
*К'
input_47         
p 

 
к "%в"
К
0         
Ъ К
H__inference_joint_model_layer_call_and_return_conditional_losses_1348962╜ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
input_46         
*К'
input_47         
p

 
к "%в"
К
0         
Ъ К
H__inference_joint_model_layer_call_and_return_conditional_losses_1349220╜ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
inputs/0         
*К'
inputs/1         
p 

 
к "%в"
К
0         
Ъ К
H__inference_joint_model_layer_call_and_return_conditional_losses_1349326╜ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
inputs/0         
*К'
inputs/1         
p

 
к "%в"
К
0         
Ъ т
-__inference_joint_model_layer_call_fn_1348474░ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
input_46         
*К'
input_47         
p 

 
к "К         т
-__inference_joint_model_layer_call_fn_1348824░ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
input_46         
*К'
input_47         
p

 
к "К         т
-__inference_joint_model_layer_call_fn_1349078░ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
inputs/0         
*К'
inputs/1         
p 

 
к "К         т
-__inference_joint_model_layer_call_fn_1349132░ "#23EFGH;<XYOPghВГКЛТУЪЫrвo
hвe
[ЪX
*К'
inputs/0         
*К'
inputs/1         
p

 
к "К         ▓
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_1349662^ЪЫ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ К
5__inference_prediction_output_0_layer_call_fn_1349652QЪЫ/в,
%в"
 К
inputs         
к "К         Ц
%__inference_signature_wrapper_1349024ь "#23EFGH;<XYOPghВГКЛТУЪЫ}вz
в 
sкp
6
input_46*К'
input_46         
6
input_47*К'
input_47         "IкF
D
prediction_output_0-К*
prediction_output_0         °
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349360вVвS
LвI
CК@
inputs4                                    
p 
к "HвE
>К;
04                                    
Ъ °
Q__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1349383вVвS
LвI
CК@
inputs4                                    
p
к "HвE
>К;
04                                    
Ъ ╨
6__inference_spatial_dropout2d_18_layer_call_fn_1349350ХVвS
LвI
CК@
inputs4                                    
p 
к ";К84                                    ╨
6__inference_spatial_dropout2d_18_layer_call_fn_1349355ХVвS
LвI
CК@
inputs4                                    
p
к ";К84                                    