Ѓ
гЖ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
ћ
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
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
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

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8оя

Adam/prediction_output_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/v

3Adam/prediction_output_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/v*
_output_shapes
:*
dtype0

!Adam/prediction_output_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/v

5Adam/prediction_output_0/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_122/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_122/bias/v
{
)Adam/dense_122/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/v*
_output_shapes
:*
dtype0

Adam/dense_122/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_122/kernel/v

+Adam/dense_122/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_121/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_121/bias/v
{
)Adam/dense_121/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_121/bias/v*
_output_shapes
:*
dtype0

Adam/dense_121/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_121/kernel/v

+Adam/dense_121/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_121/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_120/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_120/bias/v
{
)Adam/dense_120/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_120/bias/v*
_output_shapes
:*
dtype0

Adam/dense_120/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_120/kernel/v

+Adam/dense_120/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_120/kernel/v*
_output_shapes
:	*
dtype0

Adam/conv2d_242/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_242/bias/v
}
*Adam/conv2d_242/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_242/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_242/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_242/kernel/v

,Adam/conv2d_242/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_242/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_241/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_241/bias/v
}
*Adam/conv2d_241/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_241/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_241/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_241/kernel/v

,Adam/conv2d_241/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_241/kernel/v*&
_output_shapes
:	*
dtype0

Adam/conv2d_245/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_245/bias/v
}
*Adam/conv2d_245/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_245/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_245/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_245/kernel/v

,Adam/conv2d_245/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_245/kernel/v*&
_output_shapes
:*
dtype0

"Adam/batch_normalization_40/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_40/beta/v

6Adam/batch_normalization_40/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_40/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_40/gamma/v

7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/v*
_output_shapes
:*
dtype0

Adam/conv2d_244/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_244/bias/v
}
*Adam/conv2d_244/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_244/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_244/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_244/kernel/v

,Adam/conv2d_244/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_244/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_240/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_240/bias/v
}
*Adam/conv2d_240/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_240/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_240/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_240/kernel/v

,Adam/conv2d_240/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_240/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_243/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_243/bias/v
}
*Adam/conv2d_243/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_243/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_243/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_243/kernel/v

,Adam/conv2d_243/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_243/kernel/v*&
_output_shapes
:		*
dtype0

Adam/prediction_output_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/m

3Adam/prediction_output_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/m*
_output_shapes
:*
dtype0

!Adam/prediction_output_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/m

5Adam/prediction_output_0/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_122/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_122/bias/m
{
)Adam/dense_122/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/m*
_output_shapes
:*
dtype0

Adam/dense_122/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_122/kernel/m

+Adam/dense_122/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_121/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_121/bias/m
{
)Adam/dense_121/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_121/bias/m*
_output_shapes
:*
dtype0

Adam/dense_121/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_121/kernel/m

+Adam/dense_121/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_121/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_120/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_120/bias/m
{
)Adam/dense_120/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_120/bias/m*
_output_shapes
:*
dtype0

Adam/dense_120/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_120/kernel/m

+Adam/dense_120/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_120/kernel/m*
_output_shapes
:	*
dtype0

Adam/conv2d_242/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_242/bias/m
}
*Adam/conv2d_242/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_242/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_242/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_242/kernel/m

,Adam/conv2d_242/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_242/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_241/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_241/bias/m
}
*Adam/conv2d_241/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_241/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_241/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_241/kernel/m

,Adam/conv2d_241/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_241/kernel/m*&
_output_shapes
:	*
dtype0

Adam/conv2d_245/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_245/bias/m
}
*Adam/conv2d_245/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_245/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_245/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_245/kernel/m

,Adam/conv2d_245/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_245/kernel/m*&
_output_shapes
:*
dtype0

"Adam/batch_normalization_40/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_40/beta/m

6Adam/batch_normalization_40/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_40/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_40/gamma/m

7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/m*
_output_shapes
:*
dtype0

Adam/conv2d_244/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_244/bias/m
}
*Adam/conv2d_244/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_244/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_244/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_244/kernel/m

,Adam/conv2d_244/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_244/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_240/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_240/bias/m
}
*Adam/conv2d_240/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_240/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_240/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_240/kernel/m

,Adam/conv2d_240/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_240/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_243/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_243/bias/m
}
*Adam/conv2d_243/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_243/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_243/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_243/kernel/m

,Adam/conv2d_243/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_243/kernel/m*&
_output_shapes
:		*
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
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0

prediction_output_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameprediction_output_0/bias

,prediction_output_0/bias/Read/ReadVariableOpReadVariableOpprediction_output_0/bias*
_output_shapes
:*
dtype0

prediction_output_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameprediction_output_0/kernel

.prediction_output_0/kernel/Read/ReadVariableOpReadVariableOpprediction_output_0/kernel*
_output_shapes

:*
dtype0
t
dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_122/bias
m
"dense_122/bias/Read/ReadVariableOpReadVariableOpdense_122/bias*
_output_shapes
:*
dtype0
|
dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_122/kernel
u
$dense_122/kernel/Read/ReadVariableOpReadVariableOpdense_122/kernel*
_output_shapes

:*
dtype0
t
dense_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_121/bias
m
"dense_121/bias/Read/ReadVariableOpReadVariableOpdense_121/bias*
_output_shapes
:*
dtype0
|
dense_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_121/kernel
u
$dense_121/kernel/Read/ReadVariableOpReadVariableOpdense_121/kernel*
_output_shapes

:*
dtype0
t
dense_120/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_120/bias
m
"dense_120/bias/Read/ReadVariableOpReadVariableOpdense_120/bias*
_output_shapes
:*
dtype0
}
dense_120/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_120/kernel
v
$dense_120/kernel/Read/ReadVariableOpReadVariableOpdense_120/kernel*
_output_shapes
:	*
dtype0
v
conv2d_242/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_242/bias
o
#conv2d_242/bias/Read/ReadVariableOpReadVariableOpconv2d_242/bias*
_output_shapes
:*
dtype0

conv2d_242/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_242/kernel

%conv2d_242/kernel/Read/ReadVariableOpReadVariableOpconv2d_242/kernel*&
_output_shapes
:*
dtype0
v
conv2d_241/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_241/bias
o
#conv2d_241/bias/Read/ReadVariableOpReadVariableOpconv2d_241/bias*
_output_shapes
:*
dtype0

conv2d_241/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameconv2d_241/kernel

%conv2d_241/kernel/Read/ReadVariableOpReadVariableOpconv2d_241/kernel*&
_output_shapes
:	*
dtype0
v
conv2d_245/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_245/bias
o
#conv2d_245/bias/Read/ReadVariableOpReadVariableOpconv2d_245/bias*
_output_shapes
:*
dtype0

conv2d_245/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_245/kernel

%conv2d_245/kernel/Read/ReadVariableOpReadVariableOpconv2d_245/kernel*&
_output_shapes
:*
dtype0
Є
&batch_normalization_40/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_40/moving_variance

:batch_normalization_40/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_40/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_40/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_40/moving_mean

6batch_normalization_40/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_40/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_40/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_40/beta

/batch_normalization_40/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_40/beta*
_output_shapes
:*
dtype0

batch_normalization_40/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_40/gamma

0batch_normalization_40/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_40/gamma*
_output_shapes
:*
dtype0
v
conv2d_244/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_244/bias
o
#conv2d_244/bias/Read/ReadVariableOpReadVariableOpconv2d_244/bias*
_output_shapes
:*
dtype0

conv2d_244/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_244/kernel

%conv2d_244/kernel/Read/ReadVariableOpReadVariableOpconv2d_244/kernel*&
_output_shapes
:*
dtype0
v
conv2d_240/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_240/bias
o
#conv2d_240/bias/Read/ReadVariableOpReadVariableOpconv2d_240/bias*
_output_shapes
:*
dtype0

conv2d_240/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_240/kernel

%conv2d_240/kernel/Read/ReadVariableOpReadVariableOpconv2d_240/kernel*&
_output_shapes
:*
dtype0
v
conv2d_243/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_243/bias
o
#conv2d_243/bias/Read/ReadVariableOpReadVariableOpconv2d_243/bias*
_output_shapes
:*
dtype0

conv2d_243/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*"
shared_nameconv2d_243/kernel

%conv2d_243/kernel/Read/ReadVariableOpReadVariableOpconv2d_243/kernel*&
_output_shapes
:		*
dtype0

serving_default_input_81Placeholder*0
_output_shapes
:џџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџ

serving_default_input_82Placeholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
§
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_81serving_default_input_82conv2d_243/kernelconv2d_243/biasconv2d_240/kernelconv2d_240/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv2d_244/kernelconv2d_244/biasconv2d_241/kernelconv2d_241/biasconv2d_245/kernelconv2d_245/biasconv2d_242/kernelconv2d_242/biasdense_120/kerneldense_120/biasdense_121/kerneldense_121/biasdense_122/kerneldense_122/biasprediction_output_0/kernelprediction_output_0/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_4037985

NoOpNoOp
ѓ 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*­ 
valueЂ B  B 
н
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
Ш
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
Ѕ
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator* 
Ш
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
Ш
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
е
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
Ш
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op*
Ш
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op*

[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
Ш
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op*

j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 

p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
Њ
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Т
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
16
17
18
19
20
21
22
23*
В
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
14
15
16
17
18
19
20
21*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Ёtrace_0
Ђtrace_1
Ѓtrace_2
Єtrace_3* 
:
Ѕtrace_0
Іtrace_1
Їtrace_2
Јtrace_3* 
* 

Љbeta_1
Њbeta_2

Ћdecay
Ќlearning_rate
	­iter"mЈ#mЉ2mЊ3mЋ;mЌ<m­EmЎFmЏOmАPmБXmВYmГgmДhmЕ	mЖ	mЗ	mИ	mЙ	mК	mЛ	mМ	mН"vО#vП2vР3vС;vТ<vУEvФFvХOvЦPvЧXvШYvЩgvЪhvЫ	vЬ	vЭ	vЮ	vЯ	vа	vб	vв	vг*

Ўserving_default* 

"0
#1*

"0
#1*
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
a[
VARIABLE_VALUEconv2d_243/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_243/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

Лtrace_0
Мtrace_1* 

Нtrace_0
Оtrace_1* 
* 

20
31*

20
31*
* 

Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Фtrace_0* 

Хtrace_0* 
a[
VARIABLE_VALUEconv2d_240/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_240/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

;0
<1*

;0
<1*
* 

Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
a[
VARIABLE_VALUEconv2d_244/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_244/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
E0
F1
G2
H3*

E0
F1*
* 

Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

вtrace_0
гtrace_1* 

дtrace_0
еtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_40/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_40/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_40/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_40/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

O0
P1*

O0
P1*
* 

жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

лtrace_0* 

мtrace_0* 
a[
VARIABLE_VALUEconv2d_245/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_245/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

X0
Y1*

X0
Y1*
* 

нnon_trainable_variables
оlayers
пmetrics
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
VARIABLE_VALUEconv2d_241/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_241/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

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

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
№trace_0* 

ёtrace_0* 
a[
VARIABLE_VALUEconv2d_242/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_242/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

їtrace_0* 

јtrace_0* 
* 
* 
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

ўtrace_0* 

џtrace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_120/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_120/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_121/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_121/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_122/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_122/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ёtrace_0* 

Ђtrace_0* 
ke
VARIABLE_VALUEprediction_output_0/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEprediction_output_0/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*

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

Ѓ0*
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
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
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
Є	variables
Ѕ	keras_api

Іtotal

Їcount*

І0
Ї1*

Є	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_243/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_243/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_240/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_240/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_244/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_244/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_40/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_40/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_245/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_245/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_241/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_241/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_242/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_242/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_120/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_120/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_121/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_121/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_122/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_122/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/prediction_output_0/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/prediction_output_0/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_243/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_243/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_240/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_240/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_244/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_244/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_40/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_40/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_245/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_245/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_241/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_241/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_242/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_242/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_120/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_120/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_121/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_121/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_122/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_122/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/prediction_output_0/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/prediction_output_0/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_243/kernel/Read/ReadVariableOp#conv2d_243/bias/Read/ReadVariableOp%conv2d_240/kernel/Read/ReadVariableOp#conv2d_240/bias/Read/ReadVariableOp%conv2d_244/kernel/Read/ReadVariableOp#conv2d_244/bias/Read/ReadVariableOp0batch_normalization_40/gamma/Read/ReadVariableOp/batch_normalization_40/beta/Read/ReadVariableOp6batch_normalization_40/moving_mean/Read/ReadVariableOp:batch_normalization_40/moving_variance/Read/ReadVariableOp%conv2d_245/kernel/Read/ReadVariableOp#conv2d_245/bias/Read/ReadVariableOp%conv2d_241/kernel/Read/ReadVariableOp#conv2d_241/bias/Read/ReadVariableOp%conv2d_242/kernel/Read/ReadVariableOp#conv2d_242/bias/Read/ReadVariableOp$dense_120/kernel/Read/ReadVariableOp"dense_120/bias/Read/ReadVariableOp$dense_121/kernel/Read/ReadVariableOp"dense_121/bias/Read/ReadVariableOp$dense_122/kernel/Read/ReadVariableOp"dense_122/bias/Read/ReadVariableOp.prediction_output_0/kernel/Read/ReadVariableOp,prediction_output_0/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_243/kernel/m/Read/ReadVariableOp*Adam/conv2d_243/bias/m/Read/ReadVariableOp,Adam/conv2d_240/kernel/m/Read/ReadVariableOp*Adam/conv2d_240/bias/m/Read/ReadVariableOp,Adam/conv2d_244/kernel/m/Read/ReadVariableOp*Adam/conv2d_244/bias/m/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_40/beta/m/Read/ReadVariableOp,Adam/conv2d_245/kernel/m/Read/ReadVariableOp*Adam/conv2d_245/bias/m/Read/ReadVariableOp,Adam/conv2d_241/kernel/m/Read/ReadVariableOp*Adam/conv2d_241/bias/m/Read/ReadVariableOp,Adam/conv2d_242/kernel/m/Read/ReadVariableOp*Adam/conv2d_242/bias/m/Read/ReadVariableOp+Adam/dense_120/kernel/m/Read/ReadVariableOp)Adam/dense_120/bias/m/Read/ReadVariableOp+Adam/dense_121/kernel/m/Read/ReadVariableOp)Adam/dense_121/bias/m/Read/ReadVariableOp+Adam/dense_122/kernel/m/Read/ReadVariableOp)Adam/dense_122/bias/m/Read/ReadVariableOp5Adam/prediction_output_0/kernel/m/Read/ReadVariableOp3Adam/prediction_output_0/bias/m/Read/ReadVariableOp,Adam/conv2d_243/kernel/v/Read/ReadVariableOp*Adam/conv2d_243/bias/v/Read/ReadVariableOp,Adam/conv2d_240/kernel/v/Read/ReadVariableOp*Adam/conv2d_240/bias/v/Read/ReadVariableOp,Adam/conv2d_244/kernel/v/Read/ReadVariableOp*Adam/conv2d_244/bias/v/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_40/beta/v/Read/ReadVariableOp,Adam/conv2d_245/kernel/v/Read/ReadVariableOp*Adam/conv2d_245/bias/v/Read/ReadVariableOp,Adam/conv2d_241/kernel/v/Read/ReadVariableOp*Adam/conv2d_241/bias/v/Read/ReadVariableOp,Adam/conv2d_242/kernel/v/Read/ReadVariableOp*Adam/conv2d_242/bias/v/Read/ReadVariableOp+Adam/dense_120/kernel/v/Read/ReadVariableOp)Adam/dense_120/bias/v/Read/ReadVariableOp+Adam/dense_121/kernel/v/Read/ReadVariableOp)Adam/dense_121/bias/v/Read/ReadVariableOp+Adam/dense_122/kernel/v/Read/ReadVariableOp)Adam/dense_122/bias/v/Read/ReadVariableOp5Adam/prediction_output_0/kernel/v/Read/ReadVariableOp3Adam/prediction_output_0/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_4038899
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_243/kernelconv2d_243/biasconv2d_240/kernelconv2d_240/biasconv2d_244/kernelconv2d_244/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv2d_245/kernelconv2d_245/biasconv2d_241/kernelconv2d_241/biasconv2d_242/kernelconv2d_242/biasdense_120/kerneldense_120/biasdense_121/kerneldense_121/biasdense_122/kerneldense_122/biasprediction_output_0/kernelprediction_output_0/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/conv2d_243/kernel/mAdam/conv2d_243/bias/mAdam/conv2d_240/kernel/mAdam/conv2d_240/bias/mAdam/conv2d_244/kernel/mAdam/conv2d_244/bias/m#Adam/batch_normalization_40/gamma/m"Adam/batch_normalization_40/beta/mAdam/conv2d_245/kernel/mAdam/conv2d_245/bias/mAdam/conv2d_241/kernel/mAdam/conv2d_241/bias/mAdam/conv2d_242/kernel/mAdam/conv2d_242/bias/mAdam/dense_120/kernel/mAdam/dense_120/bias/mAdam/dense_121/kernel/mAdam/dense_121/bias/mAdam/dense_122/kernel/mAdam/dense_122/bias/m!Adam/prediction_output_0/kernel/mAdam/prediction_output_0/bias/mAdam/conv2d_243/kernel/vAdam/conv2d_243/bias/vAdam/conv2d_240/kernel/vAdam/conv2d_240/bias/vAdam/conv2d_244/kernel/vAdam/conv2d_244/bias/v#Adam/batch_normalization_40/gamma/v"Adam/batch_normalization_40/beta/vAdam/conv2d_245/kernel/vAdam/conv2d_245/bias/vAdam/conv2d_241/kernel/vAdam/conv2d_241/bias/vAdam/conv2d_242/kernel/vAdam/conv2d_242/bias/vAdam/dense_120/kernel/vAdam/dense_120/bias/vAdam/dense_121/kernel/vAdam/dense_121/bias/vAdam/dense_122/kernel/vAdam/dense_122/bias/v!Adam/prediction_output_0/kernel/vAdam/prediction_output_0/bias/v*W
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_4039134еќ
№
o
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038340

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н
Ђ
5__inference_prediction_output_0_layer_call_fn_4038640

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4037377o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_240_layer_call_and_return_conditional_losses_4038383

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
H
,__inference_flatten_80_layer_call_fn_4038552

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_4037305a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЬN
х
H__inference_joint_model_layer_call_and_return_conditional_losses_4037854
input_81
input_82,
conv2d_243_4037789:		 
conv2d_243_4037791:,
conv2d_240_4037794: 
conv2d_240_4037796:,
batch_normalization_40_4037800:,
batch_normalization_40_4037802:,
batch_normalization_40_4037804:,
batch_normalization_40_4037806:,
conv2d_244_4037809: 
conv2d_244_4037811:,
conv2d_241_4037814:	 
conv2d_241_4037816:,
conv2d_245_4037819: 
conv2d_245_4037821:,
conv2d_242_4037824: 
conv2d_242_4037826:$
dense_120_4037833:	
dense_120_4037835:#
dense_121_4037838:
dense_121_4037840:#
dense_122_4037843:
dense_122_4037845:-
prediction_output_0_4037848:)
prediction_output_0_4037850:
identityЂ.batch_normalization_40/StatefulPartitionedCallЂ"conv2d_240/StatefulPartitionedCallЂ"conv2d_241/StatefulPartitionedCallЂ"conv2d_242/StatefulPartitionedCallЂ"conv2d_243/StatefulPartitionedCallЂ"conv2d_244/StatefulPartitionedCallЂ"conv2d_245/StatefulPartitionedCallЂ!dense_120/StatefulPartitionedCallЂ!dense_121/StatefulPartitionedCallЂ!dense_122/StatefulPartitionedCallЂ+prediction_output_0/StatefulPartitionedCall
"conv2d_243/StatefulPartitionedCallStatefulPartitionedCallinput_82conv2d_243_4037789conv2d_243_4037791*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4037189
"conv2d_240/StatefulPartitionedCallStatefulPartitionedCallinput_81conv2d_240_4037794conv2d_240_4037796*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4037206
$spatial_dropout2d_40/PartitionedCallPartitionedCall+conv2d_243/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037061 
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_240/StatefulPartitionedCall:output:0batch_normalization_40_4037800batch_normalization_40_4037802batch_normalization_40_4037804batch_normalization_40_4037806*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037114Ў
"conv2d_244/StatefulPartitionedCallStatefulPartitionedCall-spatial_dropout2d_40/PartitionedCall:output:0conv2d_244_4037809conv2d_244_4037811*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4037233И
"conv2d_241/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_241_4037814conv2d_241_4037816*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4037250Ќ
"conv2d_245/StatefulPartitionedCallStatefulPartitionedCall+conv2d_244/StatefulPartitionedCall:output:0conv2d_245_4037819conv2d_245_4037821*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4037267Ќ
"conv2d_242/StatefulPartitionedCallStatefulPartitionedCall+conv2d_241/StatefulPartitionedCall:output:0conv2d_242_4037824conv2d_242_4037826*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4037284џ
'global_max_pooling2d_40/PartitionedCallPartitionedCall+conv2d_245/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4037166ъ
flatten_81/PartitionedCallPartitionedCall0global_max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_4037297ц
flatten_80/PartitionedCallPartitionedCall+conv2d_242/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_4037305
concatenate_40/PartitionedCallPartitionedCall#flatten_81/PartitionedCall:output:0#flatten_80/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4037314
!dense_120/StatefulPartitionedCallStatefulPartitionedCall'concatenate_40/PartitionedCall:output:0dense_120_4037833dense_120_4037835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_4037327
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_4037838dense_121_4037840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_4037344
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_4037843dense_122_4037845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_4037361Ц
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0prediction_output_0_4037848prediction_output_0_4037850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4037377
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџя
NoOpNoOp/^batch_normalization_40/StatefulPartitionedCall#^conv2d_240/StatefulPartitionedCall#^conv2d_241/StatefulPartitionedCall#^conv2d_242/StatefulPartitionedCall#^conv2d_243/StatefulPartitionedCall#^conv2d_244/StatefulPartitionedCall#^conv2d_245/StatefulPartitionedCall"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2H
"conv2d_240/StatefulPartitionedCall"conv2d_240/StatefulPartitionedCall2H
"conv2d_241/StatefulPartitionedCall"conv2d_241/StatefulPartitionedCall2H
"conv2d_242/StatefulPartitionedCall"conv2d_242/StatefulPartitionedCall2H
"conv2d_243/StatefulPartitionedCall"conv2d_243/StatefulPartitionedCall2H
"conv2d_244/StatefulPartitionedCall"conv2d_244/StatefulPartitionedCall2H
"conv2d_245/StatefulPartitionedCall"conv2d_245/StatefulPartitionedCall2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_81:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_82


G__inference_conv2d_243_layer_call_and_return_conditional_losses_4038325

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
P

H__inference_joint_model_layer_call_and_return_conditional_losses_4037680

inputs
inputs_1,
conv2d_243_4037615:		 
conv2d_243_4037617:,
conv2d_240_4037620: 
conv2d_240_4037622:,
batch_normalization_40_4037626:,
batch_normalization_40_4037628:,
batch_normalization_40_4037630:,
batch_normalization_40_4037632:,
conv2d_244_4037635: 
conv2d_244_4037637:,
conv2d_241_4037640:	 
conv2d_241_4037642:,
conv2d_245_4037645: 
conv2d_245_4037647:,
conv2d_242_4037650: 
conv2d_242_4037652:$
dense_120_4037659:	
dense_120_4037661:#
dense_121_4037664:
dense_121_4037666:#
dense_122_4037669:
dense_122_4037671:-
prediction_output_0_4037674:)
prediction_output_0_4037676:
identityЂ.batch_normalization_40/StatefulPartitionedCallЂ"conv2d_240/StatefulPartitionedCallЂ"conv2d_241/StatefulPartitionedCallЂ"conv2d_242/StatefulPartitionedCallЂ"conv2d_243/StatefulPartitionedCallЂ"conv2d_244/StatefulPartitionedCallЂ"conv2d_245/StatefulPartitionedCallЂ!dense_120/StatefulPartitionedCallЂ!dense_121/StatefulPartitionedCallЂ!dense_122/StatefulPartitionedCallЂ+prediction_output_0/StatefulPartitionedCallЂ,spatial_dropout2d_40/StatefulPartitionedCall
"conv2d_243/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_243_4037615conv2d_243_4037617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4037189
"conv2d_240/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_240_4037620conv2d_240_4037622*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4037206
,spatial_dropout2d_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_243/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037089
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_240/StatefulPartitionedCall:output:0batch_normalization_40_4037626batch_normalization_40_4037628batch_normalization_40_4037630batch_normalization_40_4037632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037145Ж
"conv2d_244/StatefulPartitionedCallStatefulPartitionedCall5spatial_dropout2d_40/StatefulPartitionedCall:output:0conv2d_244_4037635conv2d_244_4037637*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4037233И
"conv2d_241/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_241_4037640conv2d_241_4037642*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4037250Ќ
"conv2d_245/StatefulPartitionedCallStatefulPartitionedCall+conv2d_244/StatefulPartitionedCall:output:0conv2d_245_4037645conv2d_245_4037647*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4037267Ќ
"conv2d_242/StatefulPartitionedCallStatefulPartitionedCall+conv2d_241/StatefulPartitionedCall:output:0conv2d_242_4037650conv2d_242_4037652*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4037284џ
'global_max_pooling2d_40/PartitionedCallPartitionedCall+conv2d_245/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4037166ъ
flatten_81/PartitionedCallPartitionedCall0global_max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_4037297ц
flatten_80/PartitionedCallPartitionedCall+conv2d_242/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_4037305
concatenate_40/PartitionedCallPartitionedCall#flatten_81/PartitionedCall:output:0#flatten_80/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4037314
!dense_120/StatefulPartitionedCallStatefulPartitionedCall'concatenate_40/PartitionedCall:output:0dense_120_4037659dense_120_4037661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_4037327
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_4037664dense_121_4037666*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_4037344
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_4037669dense_122_4037671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_4037361Ц
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0prediction_output_0_4037674prediction_output_0_4037676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4037377
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp/^batch_normalization_40/StatefulPartitionedCall#^conv2d_240/StatefulPartitionedCall#^conv2d_241/StatefulPartitionedCall#^conv2d_242/StatefulPartitionedCall#^conv2d_243/StatefulPartitionedCall#^conv2d_244/StatefulPartitionedCall#^conv2d_245/StatefulPartitionedCall"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall-^spatial_dropout2d_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2H
"conv2d_240/StatefulPartitionedCall"conv2d_240/StatefulPartitionedCall2H
"conv2d_241/StatefulPartitionedCall"conv2d_241/StatefulPartitionedCall2H
"conv2d_242/StatefulPartitionedCall"conv2d_242/StatefulPartitionedCall2H
"conv2d_243/StatefulPartitionedCall"conv2d_243/StatefulPartitionedCall2H
"conv2d_244/StatefulPartitionedCall"conv2d_244/StatefulPartitionedCall2H
"conv2d_245/StatefulPartitionedCall"conv2d_245/StatefulPartitionedCall2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2\
,spatial_dropout2d_40/StatefulPartitionedCall,spatial_dropout2d_40/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:YU
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
е0
#__inference__traced_restore_4039134
file_prefix<
"assignvariableop_conv2d_243_kernel:		0
"assignvariableop_1_conv2d_243_bias:>
$assignvariableop_2_conv2d_240_kernel:0
"assignvariableop_3_conv2d_240_bias:>
$assignvariableop_4_conv2d_244_kernel:0
"assignvariableop_5_conv2d_244_bias:=
/assignvariableop_6_batch_normalization_40_gamma:<
.assignvariableop_7_batch_normalization_40_beta:C
5assignvariableop_8_batch_normalization_40_moving_mean:G
9assignvariableop_9_batch_normalization_40_moving_variance:?
%assignvariableop_10_conv2d_245_kernel:1
#assignvariableop_11_conv2d_245_bias:?
%assignvariableop_12_conv2d_241_kernel:	1
#assignvariableop_13_conv2d_241_bias:?
%assignvariableop_14_conv2d_242_kernel:1
#assignvariableop_15_conv2d_242_bias:7
$assignvariableop_16_dense_120_kernel:	0
"assignvariableop_17_dense_120_bias:6
$assignvariableop_18_dense_121_kernel:0
"assignvariableop_19_dense_121_bias:6
$assignvariableop_20_dense_122_kernel:0
"assignvariableop_21_dense_122_bias:@
.assignvariableop_22_prediction_output_0_kernel::
,assignvariableop_23_prediction_output_0_bias:$
assignvariableop_24_beta_1: $
assignvariableop_25_beta_2: #
assignvariableop_26_decay: +
!assignvariableop_27_learning_rate: '
assignvariableop_28_adam_iter:	 #
assignvariableop_29_total: #
assignvariableop_30_count: F
,assignvariableop_31_adam_conv2d_243_kernel_m:		8
*assignvariableop_32_adam_conv2d_243_bias_m:F
,assignvariableop_33_adam_conv2d_240_kernel_m:8
*assignvariableop_34_adam_conv2d_240_bias_m:F
,assignvariableop_35_adam_conv2d_244_kernel_m:8
*assignvariableop_36_adam_conv2d_244_bias_m:E
7assignvariableop_37_adam_batch_normalization_40_gamma_m:D
6assignvariableop_38_adam_batch_normalization_40_beta_m:F
,assignvariableop_39_adam_conv2d_245_kernel_m:8
*assignvariableop_40_adam_conv2d_245_bias_m:F
,assignvariableop_41_adam_conv2d_241_kernel_m:	8
*assignvariableop_42_adam_conv2d_241_bias_m:F
,assignvariableop_43_adam_conv2d_242_kernel_m:8
*assignvariableop_44_adam_conv2d_242_bias_m:>
+assignvariableop_45_adam_dense_120_kernel_m:	7
)assignvariableop_46_adam_dense_120_bias_m:=
+assignvariableop_47_adam_dense_121_kernel_m:7
)assignvariableop_48_adam_dense_121_bias_m:=
+assignvariableop_49_adam_dense_122_kernel_m:7
)assignvariableop_50_adam_dense_122_bias_m:G
5assignvariableop_51_adam_prediction_output_0_kernel_m:A
3assignvariableop_52_adam_prediction_output_0_bias_m:F
,assignvariableop_53_adam_conv2d_243_kernel_v:		8
*assignvariableop_54_adam_conv2d_243_bias_v:F
,assignvariableop_55_adam_conv2d_240_kernel_v:8
*assignvariableop_56_adam_conv2d_240_bias_v:F
,assignvariableop_57_adam_conv2d_244_kernel_v:8
*assignvariableop_58_adam_conv2d_244_bias_v:E
7assignvariableop_59_adam_batch_normalization_40_gamma_v:D
6assignvariableop_60_adam_batch_normalization_40_beta_v:F
,assignvariableop_61_adam_conv2d_245_kernel_v:8
*assignvariableop_62_adam_conv2d_245_bias_v:F
,assignvariableop_63_adam_conv2d_241_kernel_v:	8
*assignvariableop_64_adam_conv2d_241_bias_v:F
,assignvariableop_65_adam_conv2d_242_kernel_v:8
*assignvariableop_66_adam_conv2d_242_bias_v:>
+assignvariableop_67_adam_dense_120_kernel_v:	7
)assignvariableop_68_adam_dense_120_bias_v:=
+assignvariableop_69_adam_dense_121_kernel_v:7
)assignvariableop_70_adam_dense_121_bias_v:=
+assignvariableop_71_adam_dense_122_kernel_v:7
)assignvariableop_72_adam_dense_122_bias_v:G
5assignvariableop_73_adam_prediction_output_0_kernel_v:A
3assignvariableop_74_adam_prediction_output_0_bias_v:
identity_76ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_8ЂAssignVariableOp_9н*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0**
valueљ)Bі)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
valueЃB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_243_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_243_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_240_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_240_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_244_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_244_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_40_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_40_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_40_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_40_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_245_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_245_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_241_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_241_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_242_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_242_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp$assignvariableop_16_dense_120_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_120_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dense_121_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_121_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_122_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_122_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp.assignvariableop_22_prediction_output_0_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_prediction_output_0_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_beta_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_beta_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_decayIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp!assignvariableop_27_learning_rateIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_243_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_243_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_240_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_240_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_244_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_244_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_40_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_40_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_245_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_245_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv2d_241_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_241_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_242_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_242_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_120_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_120_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_121_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_121_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_122_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_122_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_prediction_output_0_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_prediction_output_0_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_243_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_243_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_240_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_240_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_244_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_244_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_40_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_40_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_245_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_245_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_241_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_241_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_242_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_242_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_120_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_120_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_121_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_121_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_122_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_122_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_prediction_output_0_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adam_prediction_output_0_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 С
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: Ў
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*­
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
З
c
G__inference_flatten_81_layer_call_and_return_conditional_losses_4038547

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_242_layer_call_and_return_conditional_losses_4038536

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
y

H__inference_joint_model_layer_call_and_return_conditional_losses_4038190
inputs_0
inputs_1C
)conv2d_243_conv2d_readvariableop_resource:		8
*conv2d_243_biasadd_readvariableop_resource:C
)conv2d_240_conv2d_readvariableop_resource:8
*conv2d_240_biasadd_readvariableop_resource:<
.batch_normalization_40_readvariableop_resource:>
0batch_normalization_40_readvariableop_1_resource:M
?batch_normalization_40_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_244_conv2d_readvariableop_resource:8
*conv2d_244_biasadd_readvariableop_resource:C
)conv2d_241_conv2d_readvariableop_resource:	8
*conv2d_241_biasadd_readvariableop_resource:C
)conv2d_245_conv2d_readvariableop_resource:8
*conv2d_245_biasadd_readvariableop_resource:C
)conv2d_242_conv2d_readvariableop_resource:8
*conv2d_242_biasadd_readvariableop_resource:;
(dense_120_matmul_readvariableop_resource:	7
)dense_120_biasadd_readvariableop_resource::
(dense_121_matmul_readvariableop_resource:7
)dense_121_biasadd_readvariableop_resource::
(dense_122_matmul_readvariableop_resource:7
)dense_122_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityЂ6batch_normalization_40/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_40/ReadVariableOpЂ'batch_normalization_40/ReadVariableOp_1Ђ!conv2d_240/BiasAdd/ReadVariableOpЂ conv2d_240/Conv2D/ReadVariableOpЂ!conv2d_241/BiasAdd/ReadVariableOpЂ conv2d_241/Conv2D/ReadVariableOpЂ!conv2d_242/BiasAdd/ReadVariableOpЂ conv2d_242/Conv2D/ReadVariableOpЂ!conv2d_243/BiasAdd/ReadVariableOpЂ conv2d_243/Conv2D/ReadVariableOpЂ!conv2d_244/BiasAdd/ReadVariableOpЂ conv2d_244/Conv2D/ReadVariableOpЂ!conv2d_245/BiasAdd/ReadVariableOpЂ conv2d_245/Conv2D/ReadVariableOpЂ dense_120/BiasAdd/ReadVariableOpЂdense_120/MatMul/ReadVariableOpЂ dense_121/BiasAdd/ReadVariableOpЂdense_121/MatMul/ReadVariableOpЂ dense_122/BiasAdd/ReadVariableOpЂdense_122/MatMul/ReadVariableOpЂ*prediction_output_0/BiasAdd/ReadVariableOpЂ)prediction_output_0/MatMul/ReadVariableOp
 conv2d_243/Conv2D/ReadVariableOpReadVariableOp)conv2d_243_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Г
conv2d_243/Conv2DConv2Dinputs_1(conv2d_243/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	

!conv2d_243/BiasAdd/ReadVariableOpReadVariableOp*conv2d_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_243/BiasAddBiasAddconv2d_243/Conv2D:output:0)conv2d_243/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_243/ReluReluconv2d_243/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_240/Conv2D/ReadVariableOpReadVariableOp)conv2d_240_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0В
conv2d_240/Conv2DConv2Dinputs_0(conv2d_240/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

!conv2d_240/BiasAdd/ReadVariableOpReadVariableOp*conv2d_240_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_240/BiasAddBiasAddconv2d_240/Conv2D:output:0)conv2d_240/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_240/ReluReluconv2d_240/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
spatial_dropout2d_40/IdentityIdentityconv2d_243/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ
%batch_normalization_40/ReadVariableOpReadVariableOp.batch_normalization_40_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_40/ReadVariableOp_1ReadVariableOp0batch_normalization_40_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0С
'batch_normalization_40/FusedBatchNormV3FusedBatchNormV3conv2d_240/Relu:activations:0-batch_normalization_40/ReadVariableOp:value:0/batch_normalization_40/ReadVariableOp_1:value:0>batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
 conv2d_244/Conv2D/ReadVariableOpReadVariableOp)conv2d_244_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0б
conv2d_244/Conv2DConv2D&spatial_dropout2d_40/Identity:output:0(conv2d_244/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	

!conv2d_244/BiasAdd/ReadVariableOpReadVariableOp*conv2d_244_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_244/BiasAddBiasAddconv2d_244/Conv2D:output:0)conv2d_244/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_244/ReluReluconv2d_244/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_241/Conv2D/ReadVariableOpReadVariableOp)conv2d_241_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0е
conv2d_241/Conv2DConv2D+batch_normalization_40/FusedBatchNormV3:y:0(conv2d_241/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

!conv2d_241/BiasAdd/ReadVariableOpReadVariableOp*conv2d_241_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_241/BiasAddBiasAddconv2d_241/Conv2D:output:0)conv2d_241/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_241/ReluReluconv2d_241/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_245/Conv2D/ReadVariableOpReadVariableOp)conv2d_245_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
conv2d_245/Conv2DConv2Dconv2d_244/Relu:activations:0(conv2d_245/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	

!conv2d_245/BiasAdd/ReadVariableOpReadVariableOp*conv2d_245_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_245/BiasAddBiasAddconv2d_245/Conv2D:output:0)conv2d_245/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_245/ReluReluconv2d_245/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_242/Conv2D/ReadVariableOpReadVariableOp)conv2d_242_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ч
conv2d_242/Conv2DConv2Dconv2d_241/Relu:activations:0(conv2d_242/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

!conv2d_242/BiasAdd/ReadVariableOpReadVariableOp*conv2d_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_242/BiasAddBiasAddconv2d_242/Conv2D:output:0)conv2d_242/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_242/ReluReluconv2d_242/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ~
-global_max_pooling2d_40/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ћ
global_max_pooling2d_40/MaxMaxconv2d_245/Relu:activations:06global_max_pooling2d_40/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
flatten_81/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_81/ReshapeReshape$global_max_pooling2d_40/Max:output:0flatten_81/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
flatten_80/ReshapeReshapeconv2d_242/Relu:activations:0flatten_80/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
concatenate_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :М
concatenate_40/concatConcatV2flatten_81/Reshape:output:0flatten_80/Reshape:output:0#concatenate_40/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ
dense_120/MatMul/ReadVariableOpReadVariableOp(dense_120_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_120/MatMulMatMulconcatenate_40/concat:output:0'dense_120/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_120/BiasAdd/ReadVariableOpReadVariableOp)dense_120_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_120/BiasAddBiasAdddense_120/MatMul:product:0(dense_120/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_120/ReluReludense_120/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_121/MatMul/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_121/MatMulMatMuldense_120/Relu:activations:0'dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_121/BiasAdd/ReadVariableOpReadVariableOp)dense_121_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_121/BiasAddBiasAdddense_121/MatMul:product:0(dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_121/ReluReludense_121/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_122/MatMulMatMuldense_121/Relu:activations:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ї
prediction_output_0/MatMulMatMuldense_122/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
*prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
prediction_output_0/BiasAddBiasAdd$prediction_output_0/MatMul:product:02prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџо
NoOpNoOp7^batch_normalization_40/FusedBatchNormV3/ReadVariableOp9^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_40/ReadVariableOp(^batch_normalization_40/ReadVariableOp_1"^conv2d_240/BiasAdd/ReadVariableOp!^conv2d_240/Conv2D/ReadVariableOp"^conv2d_241/BiasAdd/ReadVariableOp!^conv2d_241/Conv2D/ReadVariableOp"^conv2d_242/BiasAdd/ReadVariableOp!^conv2d_242/Conv2D/ReadVariableOp"^conv2d_243/BiasAdd/ReadVariableOp!^conv2d_243/Conv2D/ReadVariableOp"^conv2d_244/BiasAdd/ReadVariableOp!^conv2d_244/Conv2D/ReadVariableOp"^conv2d_245/BiasAdd/ReadVariableOp!^conv2d_245/Conv2D/ReadVariableOp!^dense_120/BiasAdd/ReadVariableOp ^dense_120/MatMul/ReadVariableOp!^dense_121/BiasAdd/ReadVariableOp ^dense_121/MatMul/ReadVariableOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp6batch_normalization_40/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_18batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_40/ReadVariableOp%batch_normalization_40/ReadVariableOp2R
'batch_normalization_40/ReadVariableOp_1'batch_normalization_40/ReadVariableOp_12F
!conv2d_240/BiasAdd/ReadVariableOp!conv2d_240/BiasAdd/ReadVariableOp2D
 conv2d_240/Conv2D/ReadVariableOp conv2d_240/Conv2D/ReadVariableOp2F
!conv2d_241/BiasAdd/ReadVariableOp!conv2d_241/BiasAdd/ReadVariableOp2D
 conv2d_241/Conv2D/ReadVariableOp conv2d_241/Conv2D/ReadVariableOp2F
!conv2d_242/BiasAdd/ReadVariableOp!conv2d_242/BiasAdd/ReadVariableOp2D
 conv2d_242/Conv2D/ReadVariableOp conv2d_242/Conv2D/ReadVariableOp2F
!conv2d_243/BiasAdd/ReadVariableOp!conv2d_243/BiasAdd/ReadVariableOp2D
 conv2d_243/Conv2D/ReadVariableOp conv2d_243/Conv2D/ReadVariableOp2F
!conv2d_244/BiasAdd/ReadVariableOp!conv2d_244/BiasAdd/ReadVariableOp2D
 conv2d_244/Conv2D/ReadVariableOp conv2d_244/Conv2D/ReadVariableOp2F
!conv2d_245/BiasAdd/ReadVariableOp!conv2d_245/BiasAdd/ReadVariableOp2D
 conv2d_245/Conv2D/ReadVariableOp conv2d_245/Conv2D/ReadVariableOp2D
 dense_120/BiasAdd/ReadVariableOp dense_120/BiasAdd/ReadVariableOp2B
dense_120/MatMul/ReadVariableOpdense_120/MatMul/ReadVariableOp2D
 dense_121/BiasAdd/ReadVariableOp dense_121/BiasAdd/ReadVariableOp2B
dense_121/MatMul/ReadVariableOpdense_121/MatMul/ReadVariableOp2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1


G__inference_conv2d_243_layer_call_and_return_conditional_losses_4037189

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
Ё
,__inference_conv2d_244_layer_call_fn_4038392

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4037233x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
Њ
"__inference__wrapped_model_4037052
input_81
input_82O
5joint_model_conv2d_243_conv2d_readvariableop_resource:		D
6joint_model_conv2d_243_biasadd_readvariableop_resource:O
5joint_model_conv2d_240_conv2d_readvariableop_resource:D
6joint_model_conv2d_240_biasadd_readvariableop_resource:H
:joint_model_batch_normalization_40_readvariableop_resource:J
<joint_model_batch_normalization_40_readvariableop_1_resource:Y
Kjoint_model_batch_normalization_40_fusedbatchnormv3_readvariableop_resource:[
Mjoint_model_batch_normalization_40_fusedbatchnormv3_readvariableop_1_resource:O
5joint_model_conv2d_244_conv2d_readvariableop_resource:D
6joint_model_conv2d_244_biasadd_readvariableop_resource:O
5joint_model_conv2d_241_conv2d_readvariableop_resource:	D
6joint_model_conv2d_241_biasadd_readvariableop_resource:O
5joint_model_conv2d_245_conv2d_readvariableop_resource:D
6joint_model_conv2d_245_biasadd_readvariableop_resource:O
5joint_model_conv2d_242_conv2d_readvariableop_resource:D
6joint_model_conv2d_242_biasadd_readvariableop_resource:G
4joint_model_dense_120_matmul_readvariableop_resource:	C
5joint_model_dense_120_biasadd_readvariableop_resource:F
4joint_model_dense_121_matmul_readvariableop_resource:C
5joint_model_dense_121_biasadd_readvariableop_resource:F
4joint_model_dense_122_matmul_readvariableop_resource:C
5joint_model_dense_122_biasadd_readvariableop_resource:P
>joint_model_prediction_output_0_matmul_readvariableop_resource:M
?joint_model_prediction_output_0_biasadd_readvariableop_resource:
identityЂBjoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOpЂDjoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1Ђ1joint_model/batch_normalization_40/ReadVariableOpЂ3joint_model/batch_normalization_40/ReadVariableOp_1Ђ-joint_model/conv2d_240/BiasAdd/ReadVariableOpЂ,joint_model/conv2d_240/Conv2D/ReadVariableOpЂ-joint_model/conv2d_241/BiasAdd/ReadVariableOpЂ,joint_model/conv2d_241/Conv2D/ReadVariableOpЂ-joint_model/conv2d_242/BiasAdd/ReadVariableOpЂ,joint_model/conv2d_242/Conv2D/ReadVariableOpЂ-joint_model/conv2d_243/BiasAdd/ReadVariableOpЂ,joint_model/conv2d_243/Conv2D/ReadVariableOpЂ-joint_model/conv2d_244/BiasAdd/ReadVariableOpЂ,joint_model/conv2d_244/Conv2D/ReadVariableOpЂ-joint_model/conv2d_245/BiasAdd/ReadVariableOpЂ,joint_model/conv2d_245/Conv2D/ReadVariableOpЂ,joint_model/dense_120/BiasAdd/ReadVariableOpЂ+joint_model/dense_120/MatMul/ReadVariableOpЂ,joint_model/dense_121/BiasAdd/ReadVariableOpЂ+joint_model/dense_121/MatMul/ReadVariableOpЂ,joint_model/dense_122/BiasAdd/ReadVariableOpЂ+joint_model/dense_122/MatMul/ReadVariableOpЂ6joint_model/prediction_output_0/BiasAdd/ReadVariableOpЂ5joint_model/prediction_output_0/MatMul/ReadVariableOpЊ
,joint_model/conv2d_243/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_243_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Ы
joint_model/conv2d_243/Conv2DConv2Dinput_824joint_model/conv2d_243/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
 
-joint_model/conv2d_243/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0У
joint_model/conv2d_243/BiasAddBiasAdd&joint_model/conv2d_243/Conv2D:output:05joint_model/conv2d_243/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
joint_model/conv2d_243/ReluRelu'joint_model/conv2d_243/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЊ
,joint_model/conv2d_240/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_240_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
joint_model/conv2d_240/Conv2DConv2Dinput_814joint_model/conv2d_240/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
 
-joint_model/conv2d_240/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_240_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0У
joint_model/conv2d_240/BiasAddBiasAdd&joint_model/conv2d_240/Conv2D:output:05joint_model/conv2d_240/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
joint_model/conv2d_240/ReluRelu'joint_model/conv2d_240/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
)joint_model/spatial_dropout2d_40/IdentityIdentity)joint_model/conv2d_243/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџЈ
1joint_model/batch_normalization_40/ReadVariableOpReadVariableOp:joint_model_batch_normalization_40_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
3joint_model/batch_normalization_40/ReadVariableOp_1ReadVariableOp<joint_model_batch_normalization_40_readvariableop_1_resource*
_output_shapes
:*
dtype0Ъ
Bjoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOpKjoint_model_batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ю
Djoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMjoint_model_batch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
3joint_model/batch_normalization_40/FusedBatchNormV3FusedBatchNormV3)joint_model/conv2d_240/Relu:activations:09joint_model/batch_normalization_40/ReadVariableOp:value:0;joint_model/batch_normalization_40/ReadVariableOp_1:value:0Jjoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0Ljoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Њ
,joint_model/conv2d_244/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_244_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ѕ
joint_model/conv2d_244/Conv2DConv2D2joint_model/spatial_dropout2d_40/Identity:output:04joint_model/conv2d_244/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
 
-joint_model/conv2d_244/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_244_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0У
joint_model/conv2d_244/BiasAddBiasAdd&joint_model/conv2d_244/Conv2D:output:05joint_model/conv2d_244/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
joint_model/conv2d_244/ReluRelu'joint_model/conv2d_244/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЊ
,joint_model/conv2d_241/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_241_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0љ
joint_model/conv2d_241/Conv2DConv2D7joint_model/batch_normalization_40/FusedBatchNormV3:y:04joint_model/conv2d_241/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
 
-joint_model/conv2d_241/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_241_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0У
joint_model/conv2d_241/BiasAddBiasAdd&joint_model/conv2d_241/Conv2D:output:05joint_model/conv2d_241/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
joint_model/conv2d_241/ReluRelu'joint_model/conv2d_241/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЊ
,joint_model/conv2d_245/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_245_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ь
joint_model/conv2d_245/Conv2DConv2D)joint_model/conv2d_244/Relu:activations:04joint_model/conv2d_245/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
 
-joint_model/conv2d_245/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_245_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0У
joint_model/conv2d_245/BiasAddBiasAdd&joint_model/conv2d_245/Conv2D:output:05joint_model/conv2d_245/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
joint_model/conv2d_245/ReluRelu'joint_model/conv2d_245/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЊ
,joint_model/conv2d_242/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_242_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ы
joint_model/conv2d_242/Conv2DConv2D)joint_model/conv2d_241/Relu:activations:04joint_model/conv2d_242/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
 
-joint_model/conv2d_242/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0У
joint_model/conv2d_242/BiasAddBiasAdd&joint_model/conv2d_242/Conv2D:output:05joint_model/conv2d_242/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
joint_model/conv2d_242/ReluRelu'joint_model/conv2d_242/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
9joint_model/global_max_pooling2d_40/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Я
'joint_model/global_max_pooling2d_40/MaxMax)joint_model/conv2d_245/Relu:activations:0Bjoint_model/global_max_pooling2d_40/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
joint_model/flatten_81/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Д
joint_model/flatten_81/ReshapeReshape0joint_model/global_max_pooling2d_40/Max:output:0%joint_model/flatten_81/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
joint_model/flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  Ў
joint_model/flatten_80/ReshapeReshape)joint_model/conv2d_242/Relu:activations:0%joint_model/flatten_80/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџh
&joint_model/concatenate_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ь
!joint_model/concatenate_40/concatConcatV2'joint_model/flatten_81/Reshape:output:0'joint_model/flatten_80/Reshape:output:0/joint_model/concatenate_40/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЁ
+joint_model/dense_120/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_120_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Й
joint_model/dense_120/MatMulMatMul*joint_model/concatenate_40/concat:output:03joint_model/dense_120/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,joint_model/dense_120/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_120_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
joint_model/dense_120/BiasAddBiasAdd&joint_model/dense_120/MatMul:product:04joint_model/dense_120/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
joint_model/dense_120/ReluRelu&joint_model/dense_120/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
+joint_model/dense_121/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_121_matmul_readvariableop_resource*
_output_shapes

:*
dtype0З
joint_model/dense_121/MatMulMatMul(joint_model/dense_120/Relu:activations:03joint_model/dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,joint_model/dense_121/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_121_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
joint_model/dense_121/BiasAddBiasAdd&joint_model/dense_121/MatMul:product:04joint_model/dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
joint_model/dense_121/ReluRelu&joint_model/dense_121/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
+joint_model/dense_122/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_122_matmul_readvariableop_resource*
_output_shapes

:*
dtype0З
joint_model/dense_122/MatMulMatMul(joint_model/dense_121/Relu:activations:03joint_model/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,joint_model/dense_122/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
joint_model/dense_122/BiasAddBiasAdd&joint_model/dense_122/MatMul:product:04joint_model/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
joint_model/dense_122/ReluRelu&joint_model/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџД
5joint_model/prediction_output_0/MatMul/ReadVariableOpReadVariableOp>joint_model_prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ы
&joint_model/prediction_output_0/MatMulMatMul(joint_model/dense_122/Relu:activations:0=joint_model/prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџВ
6joint_model/prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp?joint_model_prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
'joint_model/prediction_output_0/BiasAddBiasAdd0joint_model/prediction_output_0/MatMul:product:0>joint_model/prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentity0joint_model/prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџў	
NoOpNoOpC^joint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOpE^joint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12^joint_model/batch_normalization_40/ReadVariableOp4^joint_model/batch_normalization_40/ReadVariableOp_1.^joint_model/conv2d_240/BiasAdd/ReadVariableOp-^joint_model/conv2d_240/Conv2D/ReadVariableOp.^joint_model/conv2d_241/BiasAdd/ReadVariableOp-^joint_model/conv2d_241/Conv2D/ReadVariableOp.^joint_model/conv2d_242/BiasAdd/ReadVariableOp-^joint_model/conv2d_242/Conv2D/ReadVariableOp.^joint_model/conv2d_243/BiasAdd/ReadVariableOp-^joint_model/conv2d_243/Conv2D/ReadVariableOp.^joint_model/conv2d_244/BiasAdd/ReadVariableOp-^joint_model/conv2d_244/Conv2D/ReadVariableOp.^joint_model/conv2d_245/BiasAdd/ReadVariableOp-^joint_model/conv2d_245/Conv2D/ReadVariableOp-^joint_model/dense_120/BiasAdd/ReadVariableOp,^joint_model/dense_120/MatMul/ReadVariableOp-^joint_model/dense_121/BiasAdd/ReadVariableOp,^joint_model/dense_121/MatMul/ReadVariableOp-^joint_model/dense_122/BiasAdd/ReadVariableOp,^joint_model/dense_122/MatMul/ReadVariableOp7^joint_model/prediction_output_0/BiasAdd/ReadVariableOp6^joint_model/prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2
Bjoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOpBjoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp2
Djoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1Djoint_model/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12f
1joint_model/batch_normalization_40/ReadVariableOp1joint_model/batch_normalization_40/ReadVariableOp2j
3joint_model/batch_normalization_40/ReadVariableOp_13joint_model/batch_normalization_40/ReadVariableOp_12^
-joint_model/conv2d_240/BiasAdd/ReadVariableOp-joint_model/conv2d_240/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_240/Conv2D/ReadVariableOp,joint_model/conv2d_240/Conv2D/ReadVariableOp2^
-joint_model/conv2d_241/BiasAdd/ReadVariableOp-joint_model/conv2d_241/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_241/Conv2D/ReadVariableOp,joint_model/conv2d_241/Conv2D/ReadVariableOp2^
-joint_model/conv2d_242/BiasAdd/ReadVariableOp-joint_model/conv2d_242/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_242/Conv2D/ReadVariableOp,joint_model/conv2d_242/Conv2D/ReadVariableOp2^
-joint_model/conv2d_243/BiasAdd/ReadVariableOp-joint_model/conv2d_243/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_243/Conv2D/ReadVariableOp,joint_model/conv2d_243/Conv2D/ReadVariableOp2^
-joint_model/conv2d_244/BiasAdd/ReadVariableOp-joint_model/conv2d_244/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_244/Conv2D/ReadVariableOp,joint_model/conv2d_244/Conv2D/ReadVariableOp2^
-joint_model/conv2d_245/BiasAdd/ReadVariableOp-joint_model/conv2d_245/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_245/Conv2D/ReadVariableOp,joint_model/conv2d_245/Conv2D/ReadVariableOp2\
,joint_model/dense_120/BiasAdd/ReadVariableOp,joint_model/dense_120/BiasAdd/ReadVariableOp2Z
+joint_model/dense_120/MatMul/ReadVariableOp+joint_model/dense_120/MatMul/ReadVariableOp2\
,joint_model/dense_121/BiasAdd/ReadVariableOp,joint_model/dense_121/BiasAdd/ReadVariableOp2Z
+joint_model/dense_121/MatMul/ReadVariableOp+joint_model/dense_121/MatMul/ReadVariableOp2\
,joint_model/dense_122/BiasAdd/ReadVariableOp,joint_model/dense_122/BiasAdd/ReadVariableOp2Z
+joint_model/dense_122/MatMul/ReadVariableOp+joint_model/dense_122/MatMul/ReadVariableOp2p
6joint_model/prediction_output_0/BiasAdd/ReadVariableOp6joint_model/prediction_output_0/BiasAdd/ReadVariableOp2n
5joint_model/prediction_output_0/MatMul/ReadVariableOp5joint_model/prediction_output_0/MatMul/ReadVariableOp:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_81:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_82
Ю

S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037114

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_245_layer_call_and_return_conditional_losses_4037267

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_122_layer_call_and_return_conditional_losses_4038631

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
p
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4038516

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
:џџџџџџџџџџџџџџџџџџ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
w
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4038571
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
А
p
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4037166

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
:џџџџџџџџџџџџџџџџџџ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_245_layer_call_and_return_conditional_losses_4038485

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
Ё
,__inference_conv2d_240_layer_call_fn_4038372

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4037206x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы
c
G__inference_flatten_80_layer_call_and_return_conditional_losses_4038558

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё

ј
F__inference_dense_120_layer_call_and_return_conditional_losses_4038591

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
Њ
%__inference_signature_wrapper_4037985
input_81
input_82!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_81input_82unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_4037052o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_81:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_82
ї
Ё
,__inference_conv2d_241_layer_call_fn_4038494

inputs!
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4037250x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
R
6__inference_spatial_dropout2d_40_layer_call_fn_4038330

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
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037061
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ФN
у
H__inference_joint_model_layer_call_and_return_conditional_losses_4037384

inputs
inputs_1,
conv2d_243_4037190:		 
conv2d_243_4037192:,
conv2d_240_4037207: 
conv2d_240_4037209:,
batch_normalization_40_4037213:,
batch_normalization_40_4037215:,
batch_normalization_40_4037217:,
batch_normalization_40_4037219:,
conv2d_244_4037234: 
conv2d_244_4037236:,
conv2d_241_4037251:	 
conv2d_241_4037253:,
conv2d_245_4037268: 
conv2d_245_4037270:,
conv2d_242_4037285: 
conv2d_242_4037287:$
dense_120_4037328:	
dense_120_4037330:#
dense_121_4037345:
dense_121_4037347:#
dense_122_4037362:
dense_122_4037364:-
prediction_output_0_4037378:)
prediction_output_0_4037380:
identityЂ.batch_normalization_40/StatefulPartitionedCallЂ"conv2d_240/StatefulPartitionedCallЂ"conv2d_241/StatefulPartitionedCallЂ"conv2d_242/StatefulPartitionedCallЂ"conv2d_243/StatefulPartitionedCallЂ"conv2d_244/StatefulPartitionedCallЂ"conv2d_245/StatefulPartitionedCallЂ!dense_120/StatefulPartitionedCallЂ!dense_121/StatefulPartitionedCallЂ!dense_122/StatefulPartitionedCallЂ+prediction_output_0/StatefulPartitionedCall
"conv2d_243/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_243_4037190conv2d_243_4037192*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4037189
"conv2d_240/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_240_4037207conv2d_240_4037209*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4037206
$spatial_dropout2d_40/PartitionedCallPartitionedCall+conv2d_243/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037061 
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_240/StatefulPartitionedCall:output:0batch_normalization_40_4037213batch_normalization_40_4037215batch_normalization_40_4037217batch_normalization_40_4037219*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037114Ў
"conv2d_244/StatefulPartitionedCallStatefulPartitionedCall-spatial_dropout2d_40/PartitionedCall:output:0conv2d_244_4037234conv2d_244_4037236*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4037233И
"conv2d_241/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_241_4037251conv2d_241_4037253*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4037250Ќ
"conv2d_245/StatefulPartitionedCallStatefulPartitionedCall+conv2d_244/StatefulPartitionedCall:output:0conv2d_245_4037268conv2d_245_4037270*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4037267Ќ
"conv2d_242/StatefulPartitionedCallStatefulPartitionedCall+conv2d_241/StatefulPartitionedCall:output:0conv2d_242_4037285conv2d_242_4037287*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4037284џ
'global_max_pooling2d_40/PartitionedCallPartitionedCall+conv2d_245/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4037166ъ
flatten_81/PartitionedCallPartitionedCall0global_max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_4037297ц
flatten_80/PartitionedCallPartitionedCall+conv2d_242/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_4037305
concatenate_40/PartitionedCallPartitionedCall#flatten_81/PartitionedCall:output:0#flatten_80/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4037314
!dense_120/StatefulPartitionedCallStatefulPartitionedCall'concatenate_40/PartitionedCall:output:0dense_120_4037328dense_120_4037330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_4037327
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_4037345dense_121_4037347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_4037344
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_4037362dense_122_4037364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_4037361Ц
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0prediction_output_0_4037378prediction_output_0_4037380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4037377
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџя
NoOpNoOp/^batch_normalization_40/StatefulPartitionedCall#^conv2d_240/StatefulPartitionedCall#^conv2d_241/StatefulPartitionedCall#^conv2d_242/StatefulPartitionedCall#^conv2d_243/StatefulPartitionedCall#^conv2d_244/StatefulPartitionedCall#^conv2d_245/StatefulPartitionedCall"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2H
"conv2d_240/StatefulPartitionedCall"conv2d_240/StatefulPartitionedCall2H
"conv2d_241/StatefulPartitionedCall"conv2d_241/StatefulPartitionedCall2H
"conv2d_242/StatefulPartitionedCall"conv2d_242/StatefulPartitionedCall2H
"conv2d_243/StatefulPartitionedCall"conv2d_243/StatefulPartitionedCall2H
"conv2d_244/StatefulPartitionedCall"conv2d_244/StatefulPartitionedCall2H
"conv2d_245/StatefulPartitionedCall"conv2d_245/StatefulPartitionedCall2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:YU
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038465

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
г
8__inference_batch_normalization_40_layer_call_fn_4038416

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037114
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь

+__inference_dense_120_layer_call_fn_4038580

inputs
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_4037327o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_241_layer_call_and_return_conditional_losses_4038505

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы
c
G__inference_flatten_80_layer_call_and_return_conditional_losses_4037305

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038447

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№
o
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037061

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

p
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038363

inputs
identity;
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
valueB:б
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
valueB:й
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
 *   ?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ж
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:Ќ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>З
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ|
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_242_layer_call_and_return_conditional_losses_4037284

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_241_layer_call_and_return_conditional_losses_4037250

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
В
-__inference_joint_model_layer_call_fn_4038093
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ*8
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037680o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ї
Ю 
 __inference__traced_save_4038899
file_prefix0
,savev2_conv2d_243_kernel_read_readvariableop.
*savev2_conv2d_243_bias_read_readvariableop0
,savev2_conv2d_240_kernel_read_readvariableop.
*savev2_conv2d_240_bias_read_readvariableop0
,savev2_conv2d_244_kernel_read_readvariableop.
*savev2_conv2d_244_bias_read_readvariableop;
7savev2_batch_normalization_40_gamma_read_readvariableop:
6savev2_batch_normalization_40_beta_read_readvariableopA
=savev2_batch_normalization_40_moving_mean_read_readvariableopE
Asavev2_batch_normalization_40_moving_variance_read_readvariableop0
,savev2_conv2d_245_kernel_read_readvariableop.
*savev2_conv2d_245_bias_read_readvariableop0
,savev2_conv2d_241_kernel_read_readvariableop.
*savev2_conv2d_241_bias_read_readvariableop0
,savev2_conv2d_242_kernel_read_readvariableop.
*savev2_conv2d_242_bias_read_readvariableop/
+savev2_dense_120_kernel_read_readvariableop-
)savev2_dense_120_bias_read_readvariableop/
+savev2_dense_121_kernel_read_readvariableop-
)savev2_dense_121_bias_read_readvariableop/
+savev2_dense_122_kernel_read_readvariableop-
)savev2_dense_122_bias_read_readvariableop9
5savev2_prediction_output_0_kernel_read_readvariableop7
3savev2_prediction_output_0_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_243_kernel_m_read_readvariableop5
1savev2_adam_conv2d_243_bias_m_read_readvariableop7
3savev2_adam_conv2d_240_kernel_m_read_readvariableop5
1savev2_adam_conv2d_240_bias_m_read_readvariableop7
3savev2_adam_conv2d_244_kernel_m_read_readvariableop5
1savev2_adam_conv2d_244_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_m_read_readvariableop7
3savev2_adam_conv2d_245_kernel_m_read_readvariableop5
1savev2_adam_conv2d_245_bias_m_read_readvariableop7
3savev2_adam_conv2d_241_kernel_m_read_readvariableop5
1savev2_adam_conv2d_241_bias_m_read_readvariableop7
3savev2_adam_conv2d_242_kernel_m_read_readvariableop5
1savev2_adam_conv2d_242_bias_m_read_readvariableop6
2savev2_adam_dense_120_kernel_m_read_readvariableop4
0savev2_adam_dense_120_bias_m_read_readvariableop6
2savev2_adam_dense_121_kernel_m_read_readvariableop4
0savev2_adam_dense_121_bias_m_read_readvariableop6
2savev2_adam_dense_122_kernel_m_read_readvariableop4
0savev2_adam_dense_122_bias_m_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_m_read_readvariableop>
:savev2_adam_prediction_output_0_bias_m_read_readvariableop7
3savev2_adam_conv2d_243_kernel_v_read_readvariableop5
1savev2_adam_conv2d_243_bias_v_read_readvariableop7
3savev2_adam_conv2d_240_kernel_v_read_readvariableop5
1savev2_adam_conv2d_240_bias_v_read_readvariableop7
3savev2_adam_conv2d_244_kernel_v_read_readvariableop5
1savev2_adam_conv2d_244_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_v_read_readvariableop7
3savev2_adam_conv2d_245_kernel_v_read_readvariableop5
1savev2_adam_conv2d_245_bias_v_read_readvariableop7
3savev2_adam_conv2d_241_kernel_v_read_readvariableop5
1savev2_adam_conv2d_241_bias_v_read_readvariableop7
3savev2_adam_conv2d_242_kernel_v_read_readvariableop5
1savev2_adam_conv2d_242_bias_v_read_readvariableop6
2savev2_adam_dense_120_kernel_v_read_readvariableop4
0savev2_adam_dense_120_bias_v_read_readvariableop6
2savev2_adam_dense_121_kernel_v_read_readvariableop4
0savev2_adam_dense_121_bias_v_read_readvariableop6
2savev2_adam_dense_122_kernel_v_read_readvariableop4
0savev2_adam_dense_122_bias_v_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_v_read_readvariableop>
:savev2_adam_prediction_output_0_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: к*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0**
valueљ)Bі)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
valueЃB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B А
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_243_kernel_read_readvariableop*savev2_conv2d_243_bias_read_readvariableop,savev2_conv2d_240_kernel_read_readvariableop*savev2_conv2d_240_bias_read_readvariableop,savev2_conv2d_244_kernel_read_readvariableop*savev2_conv2d_244_bias_read_readvariableop7savev2_batch_normalization_40_gamma_read_readvariableop6savev2_batch_normalization_40_beta_read_readvariableop=savev2_batch_normalization_40_moving_mean_read_readvariableopAsavev2_batch_normalization_40_moving_variance_read_readvariableop,savev2_conv2d_245_kernel_read_readvariableop*savev2_conv2d_245_bias_read_readvariableop,savev2_conv2d_241_kernel_read_readvariableop*savev2_conv2d_241_bias_read_readvariableop,savev2_conv2d_242_kernel_read_readvariableop*savev2_conv2d_242_bias_read_readvariableop+savev2_dense_120_kernel_read_readvariableop)savev2_dense_120_bias_read_readvariableop+savev2_dense_121_kernel_read_readvariableop)savev2_dense_121_bias_read_readvariableop+savev2_dense_122_kernel_read_readvariableop)savev2_dense_122_bias_read_readvariableop5savev2_prediction_output_0_kernel_read_readvariableop3savev2_prediction_output_0_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_243_kernel_m_read_readvariableop1savev2_adam_conv2d_243_bias_m_read_readvariableop3savev2_adam_conv2d_240_kernel_m_read_readvariableop1savev2_adam_conv2d_240_bias_m_read_readvariableop3savev2_adam_conv2d_244_kernel_m_read_readvariableop1savev2_adam_conv2d_244_bias_m_read_readvariableop>savev2_adam_batch_normalization_40_gamma_m_read_readvariableop=savev2_adam_batch_normalization_40_beta_m_read_readvariableop3savev2_adam_conv2d_245_kernel_m_read_readvariableop1savev2_adam_conv2d_245_bias_m_read_readvariableop3savev2_adam_conv2d_241_kernel_m_read_readvariableop1savev2_adam_conv2d_241_bias_m_read_readvariableop3savev2_adam_conv2d_242_kernel_m_read_readvariableop1savev2_adam_conv2d_242_bias_m_read_readvariableop2savev2_adam_dense_120_kernel_m_read_readvariableop0savev2_adam_dense_120_bias_m_read_readvariableop2savev2_adam_dense_121_kernel_m_read_readvariableop0savev2_adam_dense_121_bias_m_read_readvariableop2savev2_adam_dense_122_kernel_m_read_readvariableop0savev2_adam_dense_122_bias_m_read_readvariableop<savev2_adam_prediction_output_0_kernel_m_read_readvariableop:savev2_adam_prediction_output_0_bias_m_read_readvariableop3savev2_adam_conv2d_243_kernel_v_read_readvariableop1savev2_adam_conv2d_243_bias_v_read_readvariableop3savev2_adam_conv2d_240_kernel_v_read_readvariableop1savev2_adam_conv2d_240_bias_v_read_readvariableop3savev2_adam_conv2d_244_kernel_v_read_readvariableop1savev2_adam_conv2d_244_bias_v_read_readvariableop>savev2_adam_batch_normalization_40_gamma_v_read_readvariableop=savev2_adam_batch_normalization_40_beta_v_read_readvariableop3savev2_adam_conv2d_245_kernel_v_read_readvariableop1savev2_adam_conv2d_245_bias_v_read_readvariableop3savev2_adam_conv2d_241_kernel_v_read_readvariableop1savev2_adam_conv2d_241_bias_v_read_readvariableop3savev2_adam_conv2d_242_kernel_v_read_readvariableop1savev2_adam_conv2d_242_bias_v_read_readvariableop2savev2_adam_dense_120_kernel_v_read_readvariableop0savev2_adam_dense_120_bias_v_read_readvariableop2savev2_adam_dense_121_kernel_v_read_readvariableop0savev2_adam_dense_121_bias_v_read_readvariableop2savev2_adam_dense_122_kernel_v_read_readvariableop0savev2_adam_dense_122_bias_v_read_readvariableop<savev2_adam_prediction_output_0_kernel_v_read_readvariableop:savev2_adam_prediction_output_0_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Ъ
_input_shapesИ
Е: :		::::::::::::	::::	:::::::: : : : : : : :		::::::::::	::::	::::::::		::::::::::	::::	:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	: 
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
:		: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:	: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::%.!

_output_shapes
:	: /
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
:		: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:	: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
::%D!

_output_shapes
:	: E
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

o
6__inference_spatial_dropout2d_40_layer_call_fn_4038335

inputs
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037089
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037145

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ё
,__inference_conv2d_245_layer_call_fn_4038474

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4037267x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_240_layer_call_and_return_conditional_losses_4037206

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_122_layer_call_and_return_conditional_losses_4037361

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г	

P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4037377

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
Ё
,__inference_conv2d_242_layer_call_fn_4038525

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4037284x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_121_layer_call_and_return_conditional_losses_4037344

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
H
,__inference_flatten_81_layer_call_fn_4038541

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_4037297`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_121_layer_call_and_return_conditional_losses_4038611

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ

+__inference_dense_121_layer_call_fn_4038600

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_4037344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
c
G__inference_flatten_81_layer_call_and_return_conditional_losses_4037297

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

U
9__inference_global_max_pooling2d_40_layer_call_fn_4038510

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4037166i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
з
В
-__inference_joint_model_layer_call_fn_4037435
input_81
input_82!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_81input_82unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_81:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_82
P

H__inference_joint_model_layer_call_and_return_conditional_losses_4037923
input_81
input_82,
conv2d_243_4037858:		 
conv2d_243_4037860:,
conv2d_240_4037863: 
conv2d_240_4037865:,
batch_normalization_40_4037869:,
batch_normalization_40_4037871:,
batch_normalization_40_4037873:,
batch_normalization_40_4037875:,
conv2d_244_4037878: 
conv2d_244_4037880:,
conv2d_241_4037883:	 
conv2d_241_4037885:,
conv2d_245_4037888: 
conv2d_245_4037890:,
conv2d_242_4037893: 
conv2d_242_4037895:$
dense_120_4037902:	
dense_120_4037904:#
dense_121_4037907:
dense_121_4037909:#
dense_122_4037912:
dense_122_4037914:-
prediction_output_0_4037917:)
prediction_output_0_4037919:
identityЂ.batch_normalization_40/StatefulPartitionedCallЂ"conv2d_240/StatefulPartitionedCallЂ"conv2d_241/StatefulPartitionedCallЂ"conv2d_242/StatefulPartitionedCallЂ"conv2d_243/StatefulPartitionedCallЂ"conv2d_244/StatefulPartitionedCallЂ"conv2d_245/StatefulPartitionedCallЂ!dense_120/StatefulPartitionedCallЂ!dense_121/StatefulPartitionedCallЂ!dense_122/StatefulPartitionedCallЂ+prediction_output_0/StatefulPartitionedCallЂ,spatial_dropout2d_40/StatefulPartitionedCall
"conv2d_243/StatefulPartitionedCallStatefulPartitionedCallinput_82conv2d_243_4037858conv2d_243_4037860*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4037189
"conv2d_240/StatefulPartitionedCallStatefulPartitionedCallinput_81conv2d_240_4037863conv2d_240_4037865*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4037206
,spatial_dropout2d_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_243/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037089
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall+conv2d_240/StatefulPartitionedCall:output:0batch_normalization_40_4037869batch_normalization_40_4037871batch_normalization_40_4037873batch_normalization_40_4037875*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037145Ж
"conv2d_244/StatefulPartitionedCallStatefulPartitionedCall5spatial_dropout2d_40/StatefulPartitionedCall:output:0conv2d_244_4037878conv2d_244_4037880*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4037233И
"conv2d_241/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_241_4037883conv2d_241_4037885*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4037250Ќ
"conv2d_245/StatefulPartitionedCallStatefulPartitionedCall+conv2d_244/StatefulPartitionedCall:output:0conv2d_245_4037888conv2d_245_4037890*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4037267Ќ
"conv2d_242/StatefulPartitionedCallStatefulPartitionedCall+conv2d_241/StatefulPartitionedCall:output:0conv2d_242_4037893conv2d_242_4037895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4037284џ
'global_max_pooling2d_40/PartitionedCallPartitionedCall+conv2d_245/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4037166ъ
flatten_81/PartitionedCallPartitionedCall0global_max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_4037297ц
flatten_80/PartitionedCallPartitionedCall+conv2d_242/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_4037305
concatenate_40/PartitionedCallPartitionedCall#flatten_81/PartitionedCall:output:0#flatten_80/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4037314
!dense_120/StatefulPartitionedCallStatefulPartitionedCall'concatenate_40/PartitionedCall:output:0dense_120_4037902dense_120_4037904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_4037327
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_4037907dense_121_4037909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_4037344
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_4037912dense_122_4037914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_4037361Ц
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0prediction_output_0_4037917prediction_output_0_4037919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4037377
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp/^batch_normalization_40/StatefulPartitionedCall#^conv2d_240/StatefulPartitionedCall#^conv2d_241/StatefulPartitionedCall#^conv2d_242/StatefulPartitionedCall#^conv2d_243/StatefulPartitionedCall#^conv2d_244/StatefulPartitionedCall#^conv2d_245/StatefulPartitionedCall"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall-^spatial_dropout2d_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2H
"conv2d_240/StatefulPartitionedCall"conv2d_240/StatefulPartitionedCall2H
"conv2d_241/StatefulPartitionedCall"conv2d_241/StatefulPartitionedCall2H
"conv2d_242/StatefulPartitionedCall"conv2d_242/StatefulPartitionedCall2H
"conv2d_243/StatefulPartitionedCall"conv2d_243/StatefulPartitionedCall2H
"conv2d_244/StatefulPartitionedCall"conv2d_244/StatefulPartitionedCall2H
"conv2d_245/StatefulPartitionedCall"conv2d_245/StatefulPartitionedCall2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2\
,spatial_dropout2d_40/StatefulPartitionedCall,spatial_dropout2d_40/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_81:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_82

p
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4037089

inputs
identity;
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
valueB:б
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
valueB:й
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
 *   ?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ж
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:Ќ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>З
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ|
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р
u
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4037314

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
г
8__inference_batch_normalization_40_layer_call_fn_4038429

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4037145
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ

+__inference_dense_122_layer_call_fn_4038620

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_4037361o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
В
-__inference_joint_model_layer_call_fn_4038039
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1


G__inference_conv2d_244_layer_call_and_return_conditional_losses_4038403

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
Ё
,__inference_conv2d_243_layer_call_fn_4038314

inputs!
unknown:		
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4037189x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г	

P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4038650

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_244_layer_call_and_return_conditional_losses_4037233

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
\
0__inference_concatenate_40_layer_call_fn_4038564
inputs_0
inputs_1
identityЧ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4037314a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
е
В
-__inference_joint_model_layer_call_fn_4037785
input_81
input_82!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_81input_82unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*8
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037680o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_81:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_82
Ё

ј
F__inference_dense_120_layer_call_and_return_conditional_losses_4037327

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
т
H__inference_joint_model_layer_call_and_return_conditional_losses_4038305
inputs_0
inputs_1C
)conv2d_243_conv2d_readvariableop_resource:		8
*conv2d_243_biasadd_readvariableop_resource:C
)conv2d_240_conv2d_readvariableop_resource:8
*conv2d_240_biasadd_readvariableop_resource:<
.batch_normalization_40_readvariableop_resource:>
0batch_normalization_40_readvariableop_1_resource:M
?batch_normalization_40_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_244_conv2d_readvariableop_resource:8
*conv2d_244_biasadd_readvariableop_resource:C
)conv2d_241_conv2d_readvariableop_resource:	8
*conv2d_241_biasadd_readvariableop_resource:C
)conv2d_245_conv2d_readvariableop_resource:8
*conv2d_245_biasadd_readvariableop_resource:C
)conv2d_242_conv2d_readvariableop_resource:8
*conv2d_242_biasadd_readvariableop_resource:;
(dense_120_matmul_readvariableop_resource:	7
)dense_120_biasadd_readvariableop_resource::
(dense_121_matmul_readvariableop_resource:7
)dense_121_biasadd_readvariableop_resource::
(dense_122_matmul_readvariableop_resource:7
)dense_122_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityЂ%batch_normalization_40/AssignNewValueЂ'batch_normalization_40/AssignNewValue_1Ђ6batch_normalization_40/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_40/ReadVariableOpЂ'batch_normalization_40/ReadVariableOp_1Ђ!conv2d_240/BiasAdd/ReadVariableOpЂ conv2d_240/Conv2D/ReadVariableOpЂ!conv2d_241/BiasAdd/ReadVariableOpЂ conv2d_241/Conv2D/ReadVariableOpЂ!conv2d_242/BiasAdd/ReadVariableOpЂ conv2d_242/Conv2D/ReadVariableOpЂ!conv2d_243/BiasAdd/ReadVariableOpЂ conv2d_243/Conv2D/ReadVariableOpЂ!conv2d_244/BiasAdd/ReadVariableOpЂ conv2d_244/Conv2D/ReadVariableOpЂ!conv2d_245/BiasAdd/ReadVariableOpЂ conv2d_245/Conv2D/ReadVariableOpЂ dense_120/BiasAdd/ReadVariableOpЂdense_120/MatMul/ReadVariableOpЂ dense_121/BiasAdd/ReadVariableOpЂdense_121/MatMul/ReadVariableOpЂ dense_122/BiasAdd/ReadVariableOpЂdense_122/MatMul/ReadVariableOpЂ*prediction_output_0/BiasAdd/ReadVariableOpЂ)prediction_output_0/MatMul/ReadVariableOp
 conv2d_243/Conv2D/ReadVariableOpReadVariableOp)conv2d_243_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Г
conv2d_243/Conv2DConv2Dinputs_1(conv2d_243/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	

!conv2d_243/BiasAdd/ReadVariableOpReadVariableOp*conv2d_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_243/BiasAddBiasAddconv2d_243/Conv2D:output:0)conv2d_243/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_243/ReluReluconv2d_243/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_240/Conv2D/ReadVariableOpReadVariableOp)conv2d_240_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0В
conv2d_240/Conv2DConv2Dinputs_0(conv2d_240/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

!conv2d_240/BiasAdd/ReadVariableOpReadVariableOp*conv2d_240_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_240/BiasAddBiasAddconv2d_240/Conv2D:output:0)conv2d_240/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_240/ReluReluconv2d_240/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџg
spatial_dropout2d_40/ShapeShapeconv2d_243/Relu:activations:0*
T0*
_output_shapes
:r
(spatial_dropout2d_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spatial_dropout2d_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spatial_dropout2d_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"spatial_dropout2d_40/strided_sliceStridedSlice#spatial_dropout2d_40/Shape:output:01spatial_dropout2d_40/strided_slice/stack:output:03spatial_dropout2d_40/strided_slice/stack_1:output:03spatial_dropout2d_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*spatial_dropout2d_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,spatial_dropout2d_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,spatial_dropout2d_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
$spatial_dropout2d_40/strided_slice_1StridedSlice#spatial_dropout2d_40/Shape:output:03spatial_dropout2d_40/strided_slice_1/stack:output:05spatial_dropout2d_40/strided_slice_1/stack_1:output:05spatial_dropout2d_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
"spatial_dropout2d_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
 spatial_dropout2d_40/dropout/MulMulconv2d_243/Relu:activations:0+spatial_dropout2d_40/dropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџu
3spatial_dropout2d_40/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
3spatial_dropout2d_40/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :П
1spatial_dropout2d_40/dropout/random_uniform/shapePack+spatial_dropout2d_40/strided_slice:output:0<spatial_dropout2d_40/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_40/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_40/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Э
9spatial_dropout2d_40/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_40/dropout/random_uniform/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0p
+spatial_dropout2d_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>э
)spatial_dropout2d_40/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_40/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_40/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџЁ
!spatial_dropout2d_40/dropout/CastCast-spatial_dropout2d_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџБ
"spatial_dropout2d_40/dropout/Mul_1Mul$spatial_dropout2d_40/dropout/Mul:z:0%spatial_dropout2d_40/dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџ
%batch_normalization_40/ReadVariableOpReadVariableOp.batch_normalization_40_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_40/ReadVariableOp_1ReadVariableOp0batch_normalization_40_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Я
'batch_normalization_40/FusedBatchNormV3FusedBatchNormV3conv2d_240/Relu:activations:0-batch_normalization_40/ReadVariableOp:value:0/batch_normalization_40/ReadVariableOp_1:value:0>batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_40/AssignNewValueAssignVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource4batch_normalization_40/FusedBatchNormV3:batch_mean:07^batch_normalization_40/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_40/AssignNewValue_1AssignVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_40/FusedBatchNormV3:batch_variance:09^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
 conv2d_244/Conv2D/ReadVariableOpReadVariableOp)conv2d_244_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0б
conv2d_244/Conv2DConv2D&spatial_dropout2d_40/dropout/Mul_1:z:0(conv2d_244/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	

!conv2d_244/BiasAdd/ReadVariableOpReadVariableOp*conv2d_244_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_244/BiasAddBiasAddconv2d_244/Conv2D:output:0)conv2d_244/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_244/ReluReluconv2d_244/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_241/Conv2D/ReadVariableOpReadVariableOp)conv2d_241_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0е
conv2d_241/Conv2DConv2D+batch_normalization_40/FusedBatchNormV3:y:0(conv2d_241/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

!conv2d_241/BiasAdd/ReadVariableOpReadVariableOp*conv2d_241_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_241/BiasAddBiasAddconv2d_241/Conv2D:output:0)conv2d_241/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_241/ReluReluconv2d_241/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_245/Conv2D/ReadVariableOpReadVariableOp)conv2d_245_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
conv2d_245/Conv2DConv2Dconv2d_244/Relu:activations:0(conv2d_245/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides	

!conv2d_245/BiasAdd/ReadVariableOpReadVariableOp*conv2d_245_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_245/BiasAddBiasAddconv2d_245/Conv2D:output:0)conv2d_245/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_245/ReluReluconv2d_245/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
 conv2d_242/Conv2D/ReadVariableOpReadVariableOp)conv2d_242_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ч
conv2d_242/Conv2DConv2Dconv2d_241/Relu:activations:0(conv2d_242/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

!conv2d_242/BiasAdd/ReadVariableOpReadVariableOp*conv2d_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_242/BiasAddBiasAddconv2d_242/Conv2D:output:0)conv2d_242/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_242/ReluReluconv2d_242/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ~
-global_max_pooling2d_40/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ћ
global_max_pooling2d_40/MaxMaxconv2d_245/Relu:activations:06global_max_pooling2d_40/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
flatten_81/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_81/ReshapeReshape$global_max_pooling2d_40/Max:output:0flatten_81/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
flatten_80/ReshapeReshapeconv2d_242/Relu:activations:0flatten_80/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
concatenate_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :М
concatenate_40/concatConcatV2flatten_81/Reshape:output:0flatten_80/Reshape:output:0#concatenate_40/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ
dense_120/MatMul/ReadVariableOpReadVariableOp(dense_120_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_120/MatMulMatMulconcatenate_40/concat:output:0'dense_120/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_120/BiasAdd/ReadVariableOpReadVariableOp)dense_120_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_120/BiasAddBiasAdddense_120/MatMul:product:0(dense_120/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_120/ReluReludense_120/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_121/MatMul/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_121/MatMulMatMuldense_120/Relu:activations:0'dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_121/BiasAdd/ReadVariableOpReadVariableOp)dense_121_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_121/BiasAddBiasAdddense_121/MatMul:product:0(dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_121/ReluReludense_121/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_122/MatMulMatMuldense_121/Relu:activations:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ї
prediction_output_0/MatMulMatMuldense_122/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
*prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
prediction_output_0/BiasAddBiasAdd$prediction_output_0/MatMul:product:02prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџА
NoOpNoOp&^batch_normalization_40/AssignNewValue(^batch_normalization_40/AssignNewValue_17^batch_normalization_40/FusedBatchNormV3/ReadVariableOp9^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_40/ReadVariableOp(^batch_normalization_40/ReadVariableOp_1"^conv2d_240/BiasAdd/ReadVariableOp!^conv2d_240/Conv2D/ReadVariableOp"^conv2d_241/BiasAdd/ReadVariableOp!^conv2d_241/Conv2D/ReadVariableOp"^conv2d_242/BiasAdd/ReadVariableOp!^conv2d_242/Conv2D/ReadVariableOp"^conv2d_243/BiasAdd/ReadVariableOp!^conv2d_243/Conv2D/ReadVariableOp"^conv2d_244/BiasAdd/ReadVariableOp!^conv2d_244/Conv2D/ReadVariableOp"^conv2d_245/BiasAdd/ReadVariableOp!^conv2d_245/Conv2D/ReadVariableOp!^dense_120/BiasAdd/ReadVariableOp ^dense_120/MatMul/ReadVariableOp!^dense_121/BiasAdd/ReadVariableOp ^dense_121/MatMul/ReadVariableOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_40/AssignNewValue%batch_normalization_40/AssignNewValue2R
'batch_normalization_40/AssignNewValue_1'batch_normalization_40/AssignNewValue_12p
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp6batch_normalization_40/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_18batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_40/ReadVariableOp%batch_normalization_40/ReadVariableOp2R
'batch_normalization_40/ReadVariableOp_1'batch_normalization_40/ReadVariableOp_12F
!conv2d_240/BiasAdd/ReadVariableOp!conv2d_240/BiasAdd/ReadVariableOp2D
 conv2d_240/Conv2D/ReadVariableOp conv2d_240/Conv2D/ReadVariableOp2F
!conv2d_241/BiasAdd/ReadVariableOp!conv2d_241/BiasAdd/ReadVariableOp2D
 conv2d_241/Conv2D/ReadVariableOp conv2d_241/Conv2D/ReadVariableOp2F
!conv2d_242/BiasAdd/ReadVariableOp!conv2d_242/BiasAdd/ReadVariableOp2D
 conv2d_242/Conv2D/ReadVariableOp conv2d_242/Conv2D/ReadVariableOp2F
!conv2d_243/BiasAdd/ReadVariableOp!conv2d_243/BiasAdd/ReadVariableOp2D
 conv2d_243/Conv2D/ReadVariableOp conv2d_243/Conv2D/ReadVariableOp2F
!conv2d_244/BiasAdd/ReadVariableOp!conv2d_244/BiasAdd/ReadVariableOp2D
 conv2d_244/Conv2D/ReadVariableOp conv2d_244/Conv2D/ReadVariableOp2F
!conv2d_245/BiasAdd/ReadVariableOp!conv2d_245/BiasAdd/ReadVariableOp2D
 conv2d_245/Conv2D/ReadVariableOp conv2d_245/Conv2D/ReadVariableOp2D
 dense_120/BiasAdd/ReadVariableOp dense_120/BiasAdd/ReadVariableOp2B
dense_120/MatMul/ReadVariableOpdense_120/MatMul/ReadVariableOp2D
 dense_121/BiasAdd/ReadVariableOp dense_121/BiasAdd/ReadVariableOp2B
dense_121/MatMul/ReadVariableOpdense_121/MatMul/ReadVariableOp2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultі
F
input_81:
serving_default_input_81:0џџџџџџџџџ
G
input_82;
serving_default_input_82:0џџџџџџџџџG
prediction_output_00
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ћ
є
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
н
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
М
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator"
_tf_keras_layer
н
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
н
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
н
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
н
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
Ѕ
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
н
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
Ѕ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
П
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
о
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
16
17
18
19
20
21
22
23"
trackable_list_wrapper
Ю
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
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
Ёtrace_0
Ђtrace_1
Ѓtrace_2
Єtrace_32ў
-__inference_joint_model_layer_call_fn_4037435
-__inference_joint_model_layer_call_fn_4038039
-__inference_joint_model_layer_call_fn_4038093
-__inference_joint_model_layer_call_fn_4037785П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0zЂtrace_1zЃtrace_2zЄtrace_3
н
Ѕtrace_0
Іtrace_1
Їtrace_2
Јtrace_32ъ
H__inference_joint_model_layer_call_and_return_conditional_losses_4038190
H__inference_joint_model_layer_call_and_return_conditional_losses_4038305
H__inference_joint_model_layer_call_and_return_conditional_losses_4037854
H__inference_joint_model_layer_call_and_return_conditional_losses_4037923П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0zІtrace_1zЇtrace_2zЈtrace_3
иBе
"__inference__wrapped_model_4037052input_81input_82"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 
Љbeta_1
Њbeta_2

Ћdecay
Ќlearning_rate
	­iter"mЈ#mЉ2mЊ3mЋ;mЌ<m­EmЎFmЏOmАPmБXmВYmГgmДhmЕ	mЖ	mЗ	mИ	mЙ	mК	mЛ	mМ	mН"vО#vП2vР3vС;vТ<vУEvФFvХOvЦPvЧXvШYvЩgvЪhvЫ	vЬ	vЭ	vЮ	vЯ	vа	vб	vв	vг"
	optimizer
-
Ўserving_default"
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
В
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ђ
Дtrace_02г
,__inference_conv2d_243_layer_call_fn_4038314Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0

Еtrace_02ю
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4038325Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
+:)		2conv2d_243/kernel
:2conv2d_243/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
с
Лtrace_0
Мtrace_12І
6__inference_spatial_dropout2d_40_layer_call_fn_4038330
6__inference_spatial_dropout2d_40_layer_call_fn_4038335Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0zМtrace_1

Нtrace_0
Оtrace_12м
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038340
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038363Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0zОtrace_1
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
В
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ђ
Фtrace_02г
,__inference_conv2d_240_layer_call_fn_4038372Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0

Хtrace_02ю
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4038383Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zХtrace_0
+:)2conv2d_240/kernel
:2conv2d_240/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
В
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
ђ
Ыtrace_02г
,__inference_conv2d_244_layer_call_fn_4038392Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЫtrace_0

Ьtrace_02ю
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4038403Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0
+:)2conv2d_244/kernel
:2conv2d_244/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
В
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
х
вtrace_0
гtrace_12Њ
8__inference_batch_normalization_40_layer_call_fn_4038416
8__inference_batch_normalization_40_layer_call_fn_4038429Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zвtrace_0zгtrace_1

дtrace_0
еtrace_12р
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038447
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038465Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0zеtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_40/gamma
):'2batch_normalization_40/beta
2:0 (2"batch_normalization_40/moving_mean
6:4 (2&batch_normalization_40/moving_variance
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
В
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ђ
лtrace_02г
,__inference_conv2d_245_layer_call_fn_4038474Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0

мtrace_02ю
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4038485Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zмtrace_0
+:)2conv2d_245/kernel
:2conv2d_245/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
В
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ђ
тtrace_02г
,__inference_conv2d_241_layer_call_fn_4038494Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zтtrace_0

уtrace_02ю
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4038505Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zуtrace_0
+:)	2conv2d_241/kernel
:2conv2d_241/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
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
џ
щtrace_02р
9__inference_global_max_pooling2d_40_layer_call_fn_4038510Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zщtrace_0

ъtrace_02ћ
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4038516Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
В
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
ђ
№trace_02г
,__inference_conv2d_242_layer_call_fn_4038525Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z№trace_0

ёtrace_02ю
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4038536Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0
+:)2conv2d_242/kernel
:2conv2d_242/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ђ
їtrace_02г
,__inference_flatten_81_layer_call_fn_4038541Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0

јtrace_02ю
G__inference_flatten_81_layer_call_and_return_conditional_losses_4038547Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
ђ
ўtrace_02г
,__inference_flatten_80_layer_call_fn_4038552Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zўtrace_0

џtrace_02ю
G__inference_flatten_80_layer_call_and_return_conditional_losses_4038558Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
і
trace_02з
0__inference_concatenate_40_layer_call_fn_4038564Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ђ
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4038571Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
+__inference_dense_120_layer_call_fn_4038580Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
F__inference_dense_120_layer_call_and_return_conditional_losses_4038591Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
#:!	2dense_120/kernel
:2dense_120/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
+__inference_dense_121_layer_call_fn_4038600Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
F__inference_dense_121_layer_call_and_return_conditional_losses_4038611Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
": 2dense_121/kernel
:2dense_121/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
+__inference_dense_122_layer_call_fn_4038620Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
F__inference_dense_122_layer_call_and_return_conditional_losses_4038631Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
": 2dense_122/kernel
:2dense_122/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ћ
Ёtrace_02м
5__inference_prediction_output_0_layer_call_fn_4038640Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0

Ђtrace_02ї
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4038650Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЂtrace_0
,:*2prediction_output_0/kernel
&:$2prediction_output_0/bias
.
G0
H1"
trackable_list_wrapper
І
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
Ѓ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_joint_model_layer_call_fn_4037435input_81input_82"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_joint_model_layer_call_fn_4038039inputs/0inputs/1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_joint_model_layer_call_fn_4038093inputs/0inputs/1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_joint_model_layer_call_fn_4037785input_81input_82"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЅBЂ
H__inference_joint_model_layer_call_and_return_conditional_losses_4038190inputs/0inputs/1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЅBЂ
H__inference_joint_model_layer_call_and_return_conditional_losses_4038305inputs/0inputs/1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЅBЂ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037854input_81input_82"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЅBЂ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037923input_81input_82"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
еBв
%__inference_signature_wrapper_4037985input_81input_82"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_conv2d_243_layer_call_fn_4038314inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4038325inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
ћBј
6__inference_spatial_dropout2d_40_layer_call_fn_4038330inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
6__inference_spatial_dropout2d_40_layer_call_fn_4038335inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038340inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038363inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_conv2d_240_layer_call_fn_4038372inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4038383inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_conv2d_244_layer_call_fn_4038392inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4038403inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
§Bњ
8__inference_batch_normalization_40_layer_call_fn_4038416inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
8__inference_batch_normalization_40_layer_call_fn_4038429inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038447inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038465inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_conv2d_245_layer_call_fn_4038474inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4038485inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_conv2d_241_layer_call_fn_4038494inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4038505inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
9__inference_global_max_pooling2d_40_layer_call_fn_4038510inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4038516inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_conv2d_242_layer_call_fn_4038525inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4038536inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_flatten_81_layer_call_fn_4038541inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_flatten_81_layer_call_and_return_conditional_losses_4038547inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
,__inference_flatten_80_layer_call_fn_4038552inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_flatten_80_layer_call_and_return_conditional_losses_4038558inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
№Bэ
0__inference_concatenate_40_layer_call_fn_4038564inputs/0inputs/1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4038571inputs/0inputs/1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
пBм
+__inference_dense_120_layer_call_fn_4038580inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_dense_120_layer_call_and_return_conditional_losses_4038591inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
пBм
+__inference_dense_121_layer_call_fn_4038600inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_dense_121_layer_call_and_return_conditional_losses_4038611inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
пBм
+__inference_dense_122_layer_call_fn_4038620inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_dense_122_layer_call_and_return_conditional_losses_4038631inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
5__inference_prediction_output_0_layer_call_fn_4038640inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4038650inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Є	variables
Ѕ	keras_api

Іtotal

Їcount"
_tf_keras_metric
0
І0
Ї1"
trackable_list_wrapper
.
Є	variables"
_generic_user_object
:  (2total
:  (2count
0:.		2Adam/conv2d_243/kernel/m
": 2Adam/conv2d_243/bias/m
0:.2Adam/conv2d_240/kernel/m
": 2Adam/conv2d_240/bias/m
0:.2Adam/conv2d_244/kernel/m
": 2Adam/conv2d_244/bias/m
/:-2#Adam/batch_normalization_40/gamma/m
.:,2"Adam/batch_normalization_40/beta/m
0:.2Adam/conv2d_245/kernel/m
": 2Adam/conv2d_245/bias/m
0:.	2Adam/conv2d_241/kernel/m
": 2Adam/conv2d_241/bias/m
0:.2Adam/conv2d_242/kernel/m
": 2Adam/conv2d_242/bias/m
(:&	2Adam/dense_120/kernel/m
!:2Adam/dense_120/bias/m
':%2Adam/dense_121/kernel/m
!:2Adam/dense_121/bias/m
':%2Adam/dense_122/kernel/m
!:2Adam/dense_122/bias/m
1:/2!Adam/prediction_output_0/kernel/m
+:)2Adam/prediction_output_0/bias/m
0:.		2Adam/conv2d_243/kernel/v
": 2Adam/conv2d_243/bias/v
0:.2Adam/conv2d_240/kernel/v
": 2Adam/conv2d_240/bias/v
0:.2Adam/conv2d_244/kernel/v
": 2Adam/conv2d_244/bias/v
/:-2#Adam/batch_normalization_40/gamma/v
.:,2"Adam/batch_normalization_40/beta/v
0:.2Adam/conv2d_245/kernel/v
": 2Adam/conv2d_245/bias/v
0:.	2Adam/conv2d_241/kernel/v
": 2Adam/conv2d_241/bias/v
0:.2Adam/conv2d_242/kernel/v
": 2Adam/conv2d_242/bias/v
(:&	2Adam/dense_120/kernel/v
!:2Adam/dense_120/bias/v
':%2Adam/dense_121/kernel/v
!:2Adam/dense_121/bias/v
':%2Adam/dense_122/kernel/v
!:2Adam/dense_122/bias/v
1:/2!Adam/prediction_output_0/kernel/v
+:)2Adam/prediction_output_0/bias/v
"__inference__wrapped_model_4037052м "#23EFGH;<XYOPghmЂj
cЂ`
^[
+(
input_81џџџџџџџџџ
,)
input_82џџџџџџџџџ
Њ "IЊF
D
prediction_output_0-*
prediction_output_0џџџџџџџџџю
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038447EFGHMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ю
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4038465EFGHMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
8__inference_batch_normalization_40_layer_call_fn_4038416EFGHMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЦ
8__inference_batch_normalization_40_layer_call_fn_4038429EFGHMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџе
K__inference_concatenate_40_layer_call_and_return_conditional_losses_4038571[ЂX
QЂN
LI
"
inputs/0џџџџџџџџџ
# 
inputs/1џџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 Ќ
0__inference_concatenate_40_layer_call_fn_4038564x[ЂX
QЂN
LI
"
inputs/0џџџџџџџџџ
# 
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЙ
G__inference_conv2d_240_layer_call_and_return_conditional_losses_4038383n238Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_240_layer_call_fn_4038372a238Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЙ
G__inference_conv2d_241_layer_call_and_return_conditional_losses_4038505nXY8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_241_layer_call_fn_4038494aXY8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЙ
G__inference_conv2d_242_layer_call_and_return_conditional_losses_4038536ngh8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_242_layer_call_fn_4038525agh8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџК
G__inference_conv2d_243_layer_call_and_return_conditional_losses_4038325o"#9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_243_layer_call_fn_4038314b"#9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЙ
G__inference_conv2d_244_layer_call_and_return_conditional_losses_4038403n;<8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_244_layer_call_fn_4038392a;<8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЙ
G__inference_conv2d_245_layer_call_and_return_conditional_losses_4038485nOP8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_245_layer_call_fn_4038474aOP8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЉ
F__inference_dense_120_layer_call_and_return_conditional_losses_4038591_0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_dense_120_layer_call_fn_4038580R0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЈ
F__inference_dense_121_layer_call_and_return_conditional_losses_4038611^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_dense_121_layer_call_fn_4038600Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЈ
F__inference_dense_122_layer_call_and_return_conditional_losses_4038631^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_dense_122_layer_call_fn_4038620Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ­
G__inference_flatten_80_layer_call_and_return_conditional_losses_4038558b8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_flatten_80_layer_call_fn_4038552U8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
G__inference_flatten_81_layer_call_and_return_conditional_losses_4038547X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 {
,__inference_flatten_81_layer_call_fn_4038541K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџн
T__inference_global_max_pooling2d_40_layer_call_and_return_conditional_losses_4038516RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Д
9__inference_global_max_pooling2d_40_layer_call_fn_4038510wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ
H__inference_joint_model_layer_call_and_return_conditional_losses_4037854Р "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
input_81џџџџџџџџџ
,)
input_82џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
H__inference_joint_model_layer_call_and_return_conditional_losses_4037923Р "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
input_81џџџџџџџџџ
,)
input_82џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
H__inference_joint_model_layer_call_and_return_conditional_losses_4038190Р "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
H__inference_joint_model_layer_call_and_return_conditional_losses_4038305Р "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 х
-__inference_joint_model_layer_call_fn_4037435Г "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
input_81џџџџџџџџџ
,)
input_82џџџџџџџџџ
p 

 
Њ "џџџџџџџџџх
-__inference_joint_model_layer_call_fn_4037785Г "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
input_81џџџџџџџџџ
,)
input_82џџџџџџџџџ
p

 
Њ "џџџџџџџџџх
-__inference_joint_model_layer_call_fn_4038039Г "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџх
-__inference_joint_model_layer_call_fn_4038093Г "#23EFGH;<XYOPghuЂr
kЂh
^[
+(
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
p

 
Њ "џџџџџџџџџВ
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4038650^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
5__inference_prediction_output_0_layer_call_fn_4038640Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
%__inference_signature_wrapper_4037985№ "#23EFGH;<XYOPghЂ}
Ђ 
vЊs
7
input_81+(
input_81џџџџџџџџџ
8
input_82,)
input_82џџџџџџџџџ"IЊF
D
prediction_output_0-*
prediction_output_0џџџџџџџџџј
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038340ЂVЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ј
Q__inference_spatial_dropout2d_40_layer_call_and_return_conditional_losses_4038363ЂVЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
6__inference_spatial_dropout2d_40_layer_call_fn_4038330VЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџа
6__inference_spatial_dropout2d_40_layer_call_fn_4038335VЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ