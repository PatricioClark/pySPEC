! Initial condition for the velocity.
! This file contains the expression used for the initial
! velocity field. You can use temporary real arrays R1-R3
! of size (1:nx,1:ny,ksta:kend) and temporary complex arrays
! C1-C6 of size (1:nz,1:ny,ista:iend) to do intermediate
! computations. The variable u0 should control the global
! amplitude of the velocity, and variables vparam0-9 can be
! used to control the amplitudes of individual terms. At the
! end, the three components of the velocity in spectral
! space should be stored in the arrays vx, vy, and vz.

! Null velocity field

!$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
            DO k = 1,nz
               vx(k,j,i) = 0.0_GP
               vy(k,j,i) = 0.0_GP
               vz(k,j,i) = 0.0_GP
            END DO
         END DO
      END DO
