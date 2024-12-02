!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!Debug pyRB!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!Save initial values of balance and initial fields
IF (cstep.ne.0) THEN
      !Save balance
      INCLUDE 'include/bouss/bouss_global.f90'

      !save files:
      WRITE(ext, fmtext) 0
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = vx(k,j,i)*rmp
               C2(k,j,i) = vy(k,j,i)*rmp
               C3(k,j,i) = vz(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
      CALL io_write(1,odir_newt,'vx',ext,planio,R1)
      CALL io_write(1,odir_newt,'vy',ext,planio,R2)
      CALL io_write(1,odir_newt,'vz',ext,planio,R3)

      ! Pressure. Transform from p'' to p to save
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*dt)
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = pr(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp2d_complex_to_real_xy(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'pr',ext,planio,R1)

      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = th(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'th',ext,planio,R1)      
ENDIF


timec = 0
timet = 0

!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!Solenoidal projection

! Apply boundary conditions and project onto solenoidal space
   !Option 1: Same as in rkstep2. Added condition in subrout such that o=-1 prevents parameter 
   ! o from modifying values 
   ! o = -1
   ! CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
   !          v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))

   ! Option 2: Only use the subroutines from sol_projection, without imposing bc. because this uses
   ! pressure field which we do not have for the modified field
   ! CALL sol_project(vx,vy,vz,pr,1,0,0)

! CALL s_imposebc(splanbc,planfc,th)
! CALL fc_filter(th)

! Hack to prevent weird accumulation in complex domain for th
! CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
! R1 = R1/nx/ny/nz
! CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

!Save check prints
! IF (cstep.ne.0) THEN
!       !Save balance
!       INCLUDE 'include/bouss/bouss_global.f90'
! ENDIF

!!!!!! End Solenoidal Projection
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Save time derivative at initial time

C18 = vx
C19 = vy
C20 = vz
C21 = th


!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz

         INCLUDE 'include/bouss/bouss_rkstep1.f90'

      END DO
   END DO
END DO

! Runge-Kutta step 2
! Evolves the system in time

DO o = ord,1,-1
   INCLUDE 'include/bouss/bouss_rkstep2.f90'
END DO

!Save time deriv in v
vx = (vx - C18)/dt
vy = (vy - C19)/dt
vz = (vz - C20)/dt
th = (th - C21)/dt

!Save deriv
IF (cstep.ne.0) THEN
      !Save balance
      INCLUDE 'include/bouss/bouss_global.f90'

      !save files:
      WRITE(ext, fmtext) 0
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = vx(k,j,i)*rmp
               C2(k,j,i) = vy(k,j,i)*rmp
               C3(k,j,i) = vz(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
      CALL io_write(1,odir_newt,'vx_deriv0',ext,planio,R1)
      CALL io_write(1,odir_newt,'vy_deriv0',ext,planio,R2)
      CALL io_write(1,odir_newt,'vz_deriv0',ext,planio,R3)

      ! Pressure. Transform from p'' to p to save
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*dt)
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = pr(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp2d_complex_to_real_xy(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'pr_deriv0',ext,planio,R1)

      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = th(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'th_deriv0',ext,planio,R1)      
ENDIF


vx = C18
vy = C19
vz = C20
th = C21

!End time derivative at initial time
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





!Evolve T time 
! step_evol = INT(T_guess/dt) !Such that if T_guess = 0 it doesn't evolve
! step_evol = ini + INT(T_guess/dt) - 1 !Such that if T_guess = 0 it doesn't evolve

dt1 = alpha*dt !smaller time step for initial steps 
T1 = N1*dt1 !evolving time using smaller time step


IF (T1.lt.T_guess) THEN
   step_evol = INT(N1 + (T_guess-T1)/dt) !Such that if T_guess = 0 it doesn't evolve
ELSE
   step_evol = INT(T_guess/dt1) !Such that if T_guess = 0 it doesn't evolve
ENDIF

IF (myrank.eq.0) THEN
OPEN(1,file='test.txt',position='append')
WRITE(1,FMT='(A,I5)') 'step_evol:', step_evol
WRITE(1,FMT='(A,es13.6)') 'T1=', T1
CLOSE(1)
ENDIF

dt_aux = dt
dt = dt1

DO t = 1,step_evol

   !Change time stepping for first N1 steps (to reduce BC error from arbitrary initial conditions)
   IF (t.gt.N1) THEN
      dt = dt_aux
   ENDIF

   timec = timec + 1
   IF (timec.eq.cstep) THEN
      INCLUDE 'include/bouss/bouss_global.f90'
      timec = 0
   ENDIF

   timet = timet + 1
   IF (timet.eq.tstep) THEN
      ! save files:
      WRITE(ext, fmtext) t-1
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = vx(k,j,i)*rmp
               C2(k,j,i) = vy(k,j,i)*rmp
               C3(k,j,i) = vz(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
      CALL io_write(1,odir_newt,'vx',ext,planio,R1)
      CALL io_write(1,odir_newt,'vy',ext,planio,R2)
      CALL io_write(1,odir_newt,'vz',ext,planio,R3)

      ! Pressure. Transform from p'' to p to save
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*dt)
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = pr(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp2d_complex_to_real_xy(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'pr',ext,planio,R1)

      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = th(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'th',ext,planio,R1)      
      timet = 0
   ENDIF

   !$omp parallel do if (iend-ista.ge.nth) private (j,k)
   DO i = ista,iend
   !$omp parallel do if (iend-ista.lt.nth) private (k)
      DO j = 1,ny
         DO k = 1,nz

            INCLUDE 'include/bouss/bouss_rkstep1.f90'

         END DO
      END DO
   END DO

   ! Runge-Kutta step 2
   ! Evolves the system in time

   DO o = ord,1,-1
      INCLUDE 'include/bouss/bouss_rkstep2.f90'
      END DO
END DO


! Translate in x direction by amount sx

! !$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
! DO i = ista,iend
!       !$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
!       DO j = 1,ny
!             DO k = 1,nz
!                   vx(k,j,i) = vx(k,j,i) * EXP( im * kx(i)*sx) 
!                   vy(k,j,i) = vy(k,j,i) * EXP( im * kx(i)*sx)
!                   vz(k,j,i) = vz(k,j,i) * EXP( im * kx(i)*sx)
!                   th(k,j,i) = th(k,j,i) * EXP( im * kx(i)*sx)
!             END DO
!       END DO
! END DO


!save files: in odir and if saving whole orbit odir newt
WRITE(ext, fmtext) 1 !always save with nmb 1 in odir

!save velocity
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = vx(k,j,i)*rmp
         C2(k,j,i) = vy(k,j,i)*rmp
         C3(k,j,i) = vz(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
CALL io_write(1,odir,'vx',ext,planio,R1)
CALL io_write(1,odir,'vy',ext,planio,R2)
CALL io_write(1,odir,'vz',ext,planio,R3)

IF (tstep.ne.0) THEN !save last fields in odir_newt
   WRITE(ext, fmtext) step_evol !last time 
   CALL io_write(1,odir_newt,'vx',ext,planio,R1)
   CALL io_write(1,odir_newt,'vy',ext,planio,R2)
   CALL io_write(1,odir_newt,'vz',ext,planio,R3)

   WRITE(ext, fmtext) 1 !rewrite for save in odir 
ENDIF

!save pressure
! Pressure. Transform from p'' to p to save
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*dt)
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = pr(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp2d_complex_to_real_xy(planfc,C1,R1,MPI_COMM_WORLD)
CALL io_write(1,odir,'pr',ext,planio,R1)

IF (tstep.ne.0) THEN !save last fields in odir_newt
   WRITE(ext, fmtext) step_evol !last time 
   CALL io_write(1,odir_newt,'pr',ext,planio,R1)
   
   WRITE(ext, fmtext) 1 !rewrite for save in odir 
ENDIF

!save th
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = th(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL io_write(1,odir,'th',ext,planio,R1)

IF (tstep.ne.0) THEN !save last fields in odir_newt
   WRITE(ext, fmtext) step_evol !last time 
   CALL io_write(1,odir_newt,'th',ext,planio,R1)
   
   WRITE(ext, fmtext) 1 !rewrite for save in odir 
ENDIF


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Save time derivative at last time

C18 = vx
C19 = vy
C20 = vz
C21 = th


!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz

         INCLUDE 'include/bouss/bouss_rkstep1.f90'

      END DO
   END DO
END DO

! Runge-Kutta step 2
! Evolves the system in time

DO o = ord,1,-1
   INCLUDE 'include/bouss/bouss_rkstep2.f90'
END DO

!Save time deriv in v
vx = (vx - C18)/dt
vy = (vy - C19)/dt
vz = (vz - C20)/dt
th = (th - C21)/dt

!Save deriv
IF (cstep.ne.0) THEN
      !Save balance
      INCLUDE 'include/bouss/bouss_global.f90'

      !save files:
      WRITE(ext, fmtext) step_evol
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = vx(k,j,i)*rmp
               C2(k,j,i) = vy(k,j,i)*rmp
               C3(k,j,i) = vz(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
      CALL io_write(1,odir_newt,'vx_derivT',ext,planio,R1)
      CALL io_write(1,odir_newt,'vy_derivT',ext,planio,R2)
      CALL io_write(1,odir_newt,'vz_derivT',ext,planio,R3)

      ! Pressure. Transform from p'' to p to save
      rmp = 1.0_GP/ &
         (real(nx,kind=GP)*real(ny,kind=GP)*dt)
      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = pr(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp2d_complex_to_real_xy(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'pr_derivT',ext,planio,R1)

      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               C1(k,j,i) = th(k,j,i)*rmp
            END DO
         END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL io_write(1,odir_newt,'th_derivT',ext,planio,R1)      
ENDIF

!End time derivative at last time
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
