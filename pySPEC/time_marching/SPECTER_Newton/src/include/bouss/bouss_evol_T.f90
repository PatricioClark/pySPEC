!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!Evolve system in T time!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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

ENDIF


timec = 0
timet = 0

!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!Solenoidal projection

! Apply boundary conditions and project onto solenoidal space
   !Option 1: Same as in rkstep2. Added condition in subrout such that o=-1 prevents parameter 
   ! o from modifying values 
   o = -1
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
IF (cstep.ne.0) THEN
      !Save balance
      INCLUDE 'include/bouss/bouss_global.f90'
ENDIF

!!!!!! End Solenoidal Projection
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!Evolve T time 

dt1 = alpha*dt !smaller time step for initial steps 
T1 = N1*dt1 !evolving time using smaller time step


IF (T1.lt.T_guess) THEN
   step_evol = INT(N1 + (T_guess-T1)/dt) + 1 !Such that if T_guess = 0 it doesn't evolve (+1 for last variable step)
ELSE
   step_evol = INT(T_guess/dt1) + 1 !Such that if T_guess = 0 it doesn't evolve
ENDIF

IF (myrank.eq.0) THEN
OPEN(1,file='test_dt.txt',position='append')
WRITE(1,*) step_evol,  T_guess !, N1, T1
CLOSE(1)
ENDIF

dt_aux = dt
dt = dt1

tind = 0
!start evol for period
DO t = 1,step_evol

   !Change time stepping for first N1 steps (to reduce BC error from arbitrary initial conditions)
   IF (t.gt.N1) THEN
      dt = dt_aux
   ENDIF

   !Change time stepping for last step to complete the full period
   IF (t.eq.step_evol) THEN
      dt = T_guess - (step_evol-1) * dt

      IF (myrank.eq.0) THEN
      OPEN(1,file='test_dt.txt',position='append')
      WRITE(1,FMT='(E12.4)')  dt
      CLOSE(1)
      ENDIF

   ENDIF

   !Prevents last step if T_guess is divisible by dt that doesn't evolve but projects (in rkstep2)
   IF (dt.lt.1e-8) THEN
      CYCLE
   ENDIF

   timec = timec + 1
   IF (timec.eq.cstep) THEN
      INCLUDE 'include/bouss/bouss_global.f90'
      timec = 0
   ENDIF

   timet = timet + 1
   ! save files:
   IF (timet.eq.tstep) THEN
      tind = tind+1
      WRITE(ext, fmtext) tind
         
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
   IF (t.eq.1) THEN
      DO o = ord,1,-1
      ! Step 2 of Runge-Kutta for the HD equations
      ! Computes the nonlinear terms and evolves the equations in dt/o
               rmp = 1/real(o, kind=GP)

               ! Non-linear terms
               CALL gradre(vx,vy,vz,C4,C5,C6)
               CALL advect(vx,vy,vz,th,C8)

      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
               DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
                  DO j = 1,ny
                     DO k = 1,nz
                        C6(k,j,i) = C6(k,j,i) - xmom*th(k,j,i)  ! Bouyancy
                        C8(k,j,i) = C8(k,j,i) - xtemp*vz(k,j,i) ! Heat current
                     ENDDO
                  ENDDO
               ENDDO

               ! Dealias non-linear 
               CALL fc_filter(C4)
               CALL fc_filter(C5)
               CALL fc_filter(C6)
               CALL fc_filter(C8)

               ! Laplacian
               CALL laplak(vx,vx)
               CALL laplak(vy,vy)
               CALL laplak(vz,vz)
               CALL laplak(th,th)

      !$omp parallel do if (iend-ista.ge.nth) private (j,k)
               DO i = ista,iend
      !$omp parallel do if (iend-ista.lt.nth) private (k)
               DO j = 1,ny
               DO k = 1,nz
                  ! Perform half oth-RK step
                  vx(k,j,i) = C1(k,j,i) + dt*(nu*vx(k,j,i)-C4(k,j,i)+&
                                          fx(k,j,i))*rmp
                  vy(k,j,i) = C2(k,j,i) + dt*(nu*vy(k,j,i)-C5(k,j,i)+&
                                          fy(k,j,i))*rmp
                  vz(k,j,i) = C3(k,j,i) + dt*(nu*vz(k,j,i)-C6(k,j,i)+&
                                          fz(k,j,i))*rmp
                  th(k,j,i) = C7(k,j,i) + dt*(kappa*th(k,j,i)-C8(k,j,i)+&
                                          fs(k,j,i))*rmp
               ENDDO
               ENDDO
               ENDDO

         CALL sol_project(vx,vy,vz,pr,1,0,0)
         ! Apply boundary conditions and project onto solenoidal space
         ! CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
               !   v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))   

         CALL s_imposebc(splanbc,planfc,th)
         CALL fc_filter(th)

         ! Hack to prevent weird accumulation in complex domain for th
         CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
         R1 = R1/nx/ny/nz
         CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

      END DO

   ELSE
   DO o = ord,1,-1
      INCLUDE 'include/bouss/bouss_rkstep2.f90'
   END DO
   ENDIF

END DO

!Print final balance:
IF (cstep.gt.0) THEN
   INCLUDE 'include/bouss/bouss_global.f90'
ENDIF


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
   tind = tind+1
   WRITE(ext, fmtext) tind
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
   WRITE(ext, fmtext) tind
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
   WRITE(ext, fmtext) tind
   CALL io_write(1,odir_newt,'th',ext,planio,R1)
   
   WRITE(ext, fmtext) 1 !rewrite for save in odir 
ENDIF
