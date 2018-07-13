create table cameras
(
    cameraid    integer NOT NULL UNIQUE AUTO_INCREMENT,
    param1      varchar(40) NOT NULL,
    param2      varchar(40) NOT NULL,
    param3      varchar(40) NOT NULL,
    x   varchar(10) NOT NULL,
    y   varchar(10) NOT NULL,
    areaid      integer NOT NULL,
    primary key (cameraid)
);

create table history
(
    historyid  integer NOT NULL UNIQUE AUTO_INCREMENT,
    cameraid integer NOT NULL,
    areaid integer NOT NULL,
    filename varchar(100) NOT NULL,
    primary key (historyid)
);

create table map
(
    areaid integer NOT NULL UNIQUE AUTO_INCREMENT,
    map text,
    primary key (areaid)
)