      implicit none
      real*8 time(1000000),Ip1(10000000),Ip2(1000000)
      real*8 diamag(10000000),micro1(10000000),micro2(1000000)
      real*8 diamag_max,time_at_dagmax,Ip1_at_dagmax,Ip2_at_dagmax
      real*8 micro1_at_dagmax,micro2_at_dagmax
      
      real*8 starttime,endtime
      
      integer imax,i
      character tab
      character*5 cshotno
      character*5 cstarttime,cendtime

      real*8 time_a(1000),Wp_a(1000),ne_a(1000)
      character*5 shot_a(1000)
      integer kk
      tab = char(9)

c      write(6,*) 'Enter Start time.'
c      read(5,*) starttime
c      write(6,*) 'Enter End time.'
c      read(5,*) endtime
      
      call getarg(1,cshotno)
      call getarg(2,cstarttime)
      call getarg(3,cendtime)
c
      write(6,*) cshotno
c
      read(cstarttime,*) starttime
      read(cendtime,*) endtime
c
      write(6,*) cstarttime
      write(6,*) cendtime
c      write(6,*) starttime
c      write(6,*) endtime

      open(10,file =cshotno//"curdiamic.txt")

      do 20 i = 1 ,10000000
         read(10,*,end = 25) time(i),Ip1(i),Ip2(i),diamag(i),
     &                                         micro1(i),micro2(i)
 20   continue
 25   close(10)

      imax = i - 1 
      diamag_max= 0d0
      do 30 i = 1 ,imax
         if((diamag(i).GT.diamag_max).and.((time(i).GT.starttime)
     &.and.(time(i).LT.endtime))) then
            time_at_dagmax = time(i)
            Ip1_at_dagmax = Ip1(i)
            Ip2_at_dagmax = Ip2(i)
            diamag_max = diamag(i)
            micro1_at_dagmax = micro1(i)
            micro2_at_dagmax = micro2(i)
         endif
 30   continue
      write(*,500) "ok?"

      write(*,500) cshotno,"  time@dagmax=",time_at_dagmax,
     &                       "  diamag_max=",diamag_max,
     &                       "  Ip@dagmax=",Ip1_at_dagmax,
     &                         "  micro@dagmax=",micro1_at_dagmax
 500  format(a,a,f6.1,4(a,f6.3))

c      open(40,file ="shot-time-dia-mic1.txt")
c      do 50 i = 1 ,1000
c         read(10,*,end = 45) shot_a(i),time_a(i),Wp_a(i),ne_a(i)
c         read(10,*,end = 45) time_a(i),Wp_a(i),ne_a(i),kk
c         kk=i
c         write(*,*) kk,Wp_a(i),ne_a(i)
c 50   continue
c 45   close(40)
c      write(*,*) 'ok1'

      open(60,file ="shot-time-dia-mic.txt",access='append')
c      do 70 i = 1 ,kk
c      write(*,*) i,kk
c      write(*,*) 'ok2'
c      write(60,*) shot_a(i),time_a(i),Wp_a(i),ne_a(i)
c 70   continue
      write(60,*) cshotno,time_at_dagmax,
     &                      diamag_max,
     &                      micro1_at_dagmax
      close(60)
      stop
      end
