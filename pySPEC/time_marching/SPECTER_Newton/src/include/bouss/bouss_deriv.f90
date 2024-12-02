!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!Calculate time derivative!!!!!!!!!!!!!!!
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

!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz

         INCLUDE 'include/bouss/bouss_rkstep1.f90'

      END DO
   END DO
END DO

! ! Runge-Kutta step 2
! ! Evolves the system in time

DO o = ord,1,-1
   INCLUDE 'include/bouss/bouss_rkstep2.f90'
END DO

!Print final balance:
IF (cstep.gt.0) THEN
   INCLUDE 'include/bouss/bouss_global.f90'
ENDIF


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
tind = tind+1
WRITE(ext, fmtext) tind
CALL io_write(1,odir_newt,'vx',ext,planio,R1)
CALL io_write(1,odir_newt,'vy',ext,planio,R2)
CALL io_write(1,odir_newt,'vz',ext,planio,R3)

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
CALL io_write(1,odir_newt,'pr',ext,planio,R1)

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
WRITE(ext, fmtext) tind
CALL io_write(1,odir_newt,'th',ext,planio,R1)

!!!!!!!!!!!!!!!!!!!
!Make null last step (but still projects solenoidally)
!!!!!!!!!!!!!!!!!!!
dt = 0 

!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz

         INCLUDE 'include/bouss/bouss_rkstep1.f90'

      END DO
   END DO
END DO

! ! Runge-Kutta step 2
! ! Evolves the system in time

DO o = ord,1,-1
   INCLUDE 'include/bouss/bouss_rkstep2.f90'
END DO

!Print final balance:
IF (cstep.gt.0) THEN
   INCLUDE 'include/bouss/bouss_global.f90'
ENDIF


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
tind = tind+1
WRITE(ext, fmtext) tind
CALL io_write(1,odir_newt,'vx',ext,planio,R1)
CALL io_write(1,odir_newt,'vy',ext,planio,R2)
CALL io_write(1,odir_newt,'vz',ext,planio,R3)

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
CALL io_write(1,odir_newt,'pr',ext,planio,R1)

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
WRITE(ext, fmtext) tind
CALL io_write(1,odir_newt,'th',ext,planio,R1)

