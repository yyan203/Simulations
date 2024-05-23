# 2.5d Lennard-Jones  indentation simulation
variable	relaxtime equal 10000
variable	myR  	  equal 100  # indenter tip radius 
variable	myA  	  equal 30 # indenter tip angle 
variable	cof  	  equal 0.05  # coefficient of friction of the indenter
print 		"remember change relaxation time!"
print 		"tip radius = ${myR}"
print 		"tip angle = ${myA}" # not semi-angle
units		lj
atom_style	atomic
dimension	3
boundary 	p p p

read_data	./wa.sample-0.375-1.readdata

variable 	xx equal 40
variable 	yy equal 24
replicate	${xx} ${yy} 1

change_box	all boundary p s p units box

mass		1 2.0
mass		2 1.0

variable	a equal 1.5
variable	b equal 0.375
variable	t equal 0.6
variable	w equal 0.022
variable	p equal 2.0
variable	l equal 28.69*6/3.0
variable	z equal $l/6.0

pair_style	lj/cut/bump 2.5
pair_coeff	1 1 1.0 1.0 $a 1.0 $b 2.5
pair_coeff	1 2 1.0 0.9166666666667 $a 1.0 $b 2.2916667
pair_coeff	2 2 1.0 0.8333333333333 $a 1.0 $b 2.0833331

pair_modify	shift yes tail no
neighbor	0.5 bin
neigh_modify	every 20 delay 0 check no
timestep	0.01

reset_timestep  0
variable	tip_h equal yhi*5.0/6-3.0
variable	boundary equal yhi*5.0/6.0
print		"boundary= ${boundary}"
variable	tip_r equal ${myR} 
variable	tip_x equal xhi/2.0 
variable	tip_y equal ${boundary}+${tip_r}
variable	tip_angle equal ${myA}   # semi-angle of the indenter
variable	tip_z equal zhi/2
variable	trunc equal ${boundary}
region		rtop block INF INF ${trunc} INF INF INF units box
delete_atoms 	region rtop
region		rmdn block INF INF INF 2.000000 INF INF units box
group		gmdn region rmdn

variable        g equal (xhi-xlo)/15.0 
change_box all x delta -$g $g y delta -2 0 boundary f s p units box
################## Relaxation  ########################
fix             1 all nvt temp $w $w 1.0 
region		rdn block INF INF INF 0.0 INF INF units box
delete_atoms 	region rdn 
fix             200 all wall/lj126 ylo -1.0 1.0 1.0 2.5 units box
thermo_style    custom step temp pe etotal vol press f_200[1] 
thermo		100
thermo_modify   lost ignore flush yes norm no
run	 	${relaxtime}	
################## Indent ########################
reset_timestep  0
variable        rad equal ${tip_r} 
print		"rad= ${rad}"
variable        k equal 1000/1.12     #indent rate[A/40fs]
variable	mvel equal 30 
variable	svel equal ${mvel}/540 
variable        y equal ${tip_y}+2.5-step*dt*${mvel}/540 
variable        disp1 equal  -1*step*dt*${mvel}/540
variable        halfx equal (xhi+xlo)/2 
fix 		100 all indent $k triangle z v_halfx v_y ${rad} angle ${tip_angle} orient y minus side out frict ${cof} approach ${svel} units box 
thermo_style    custom step temp pe etotal vol press f_100[2] f_200[1] v_disp1
thermo_modify   lost ignore flush yes norm no
thermo		100
variable	ratio equal 0.20
variable	runstep equal ceil(${trunc}*${ratio}*54/${mvel})*1000
print		"runstep= ${runstep}"
variable	dumpf equal ${runstep}/20
variable	dumpf2 equal ${runstep}/40
variable	zthick equal zhi-zlo+0.01 
variable	dumpf3 equal ${runstep}/5
restart 	${dumpf3} back.restart.*
variable        dumpf4 equal ${dumpf}*2
fix             0 all balance ${dumpf4} 1.05 shift xy 10 1.05 out gridinfo
dump            4 all custom ${dumpf} lata4olivia_indent.* id type x y z vx vy vz fx fy fz
compute		p all stress/atom NULL
fix		6 all ave/spatial ${dumpf2} 1 ${dumpf2} x center 5.0 y 0.0 5.0 z lower ${zthick} c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6] units box file indent-stress-dist ave one
run             ${runstep} 
################## unloading ########################
variable	cy equal v_y 
variable	cdis equal v_disp1 
print		"current disp1= ${cdis}"
variable	cstep equal step 
print		"current step= ${cstep}"
variable        y2 equal ${cy}+(step-${cstep})*dt*${mvel}/540 
variable        disp1 equal (step-${cstep})*dt*${mvel}/540+${cdis}
unfix		100
fix 		100 all indent $k triangle z v_halfx v_y2 ${rad} angle ${tip_angle} orient y minus side out frict ${cof} retract ${svel} units box
variable	temp equal ${runstep}/4
variable	unloadstep equal ceil(${temp})
print		"unloadstep= ${unloadstep}"
run             ${unloadstep} 
