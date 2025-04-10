graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1047
  arrival_time 22132.400840538776
  lifetime 1136.8906213429482
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 45
    gpu 38
    rom 39
  ]
  node [
    id 1
    label "1"
    cpu 15
    gpu 3
    rom 46
  ]
  node [
    id 2
    label "2"
    cpu 24
    gpu 6
    rom 21
  ]
  node [
    id 3
    label "3"
    cpu 43
    gpu 42
    rom 11
  ]
  node [
    id 4
    label "4"
    cpu 21
    gpu 18
    rom 43
  ]
  node [
    id 5
    label "5"
    cpu 38
    gpu 42
    rom 31
  ]
  node [
    id 6
    label "6"
    cpu 41
    gpu 8
    rom 21
  ]
  node [
    id 7
    label "7"
    cpu 13
    gpu 37
    rom 45
  ]
  node [
    id 8
    label "8"
    cpu 42
    gpu 48
    rom 32
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 33
    rom 46
  ]
  node [
    id 10
    label "10"
    cpu 1
    gpu 32
    rom 13
  ]
  node [
    id 11
    label "11"
    cpu 41
    gpu 28
    rom 9
  ]
  node [
    id 12
    label "12"
    cpu 37
    gpu 48
    rom 27
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 22
  ]
  edge [
    source 3
    target 4
    bw 11
  ]
  edge [
    source 4
    target 5
    bw 14
  ]
  edge [
    source 5
    target 6
    bw 38
  ]
  edge [
    source 6
    target 7
    bw 29
  ]
  edge [
    source 7
    target 8
    bw 2
  ]
  edge [
    source 8
    target 9
    bw 31
  ]
  edge [
    source 9
    target 10
    bw 10
  ]
  edge [
    source 10
    target 11
    bw 42
  ]
  edge [
    source 11
    target 12
    bw 35
  ]
]
